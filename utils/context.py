# Modified from https://github.com/finegrain-ai/finegrain-python/blob/eaa3cb5a77a889a8f738e4f35c780399bfc3e8a9/finegrain/src/finegrain/__init__.py

import asyncio
import configparser
import dataclasses as dc
import io
import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, BinaryIO, Literal, NewType, cast, get_args

import httpx
import httpx_sse
from httpx._types import QueryParamTypes, RequestData, RequestFiles
from PIL import Image

logger = logging.getLogger(__name__)

Priority = Literal["low", "standard", "high"]
StateID = NewType("StateID", str)

VERSION = "0.1"

API_KEY_PATTERN = re.compile(r"^FGAPI(\-[A-Z0-9]{6}){4}$")
EMAIL_PWD_PATTERN = re.compile(r"^\s*(?P<email>[\S]+@[\S]+):(?P<pwd>\S+)\s*$")


def check_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        msg = f"{e!s}\nBody: {response.text}"
        exc = httpx.HTTPStatusError(message=msg, request=e.request, response=e.response)
        raise exc from e


class SSELoopStopped(RuntimeError):
    first_error: Exception | None
    last_error: Exception | None

    def __init__(
        self,
        message: str | None = None,
        first_error: Exception | None = None,
        last_error: Exception | None = None,
    ) -> None:
        self.first_error = first_error
        self.last_error = last_error
        super().__init__(message or self.default_message)

    @property
    def default_message(self) -> str:
        return f"SSE loop stopped (first error: {self.first_error}, last error: {self.last_error})"


class Futures[Tk, Tv]:
    _event_loop: asyncio.AbstractEventLoop | None

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()
        else:
            assert self._event_loop == asyncio.get_running_loop(), "event loop changed"
        return self._event_loop

    def create_future(self) -> asyncio.Future[Tv]:
        return self.event_loop.create_future()

    def __init__(self, capacity: int = 256) -> None:
        self.futures = defaultdict[Tk, asyncio.Future[Tv]](self.create_future)
        self.capacity = capacity
        self._event_loop = None

    def cull(self) -> None:
        while len(self.futures) >= self.capacity:
            del self.futures[next(iter(self.futures))]

    def __getitem__(self, key: Tk) -> asyncio.Future[Tv]:
        assert self.event_loop
        self.cull()
        return self.futures[key]

    def __delitem__(self, key: Tk) -> None:
        try:
            del self.futures[key]
        except KeyError:
            pass


class RetryContext:
    max_failures: int
    max_jitter: float
    max_backoff: float
    exp_base: float
    exp_factor: float

    failures: int
    first_error: Exception | None
    last_error: Exception | None

    def __init__(
        self,
        max_failures: int = 10,
        max_jitter: float = 1.0,
        max_backoff: float = 15.0,
        exp_base: float = 2.0,
        exp_factor: float = 0.1,
    ):
        self.max_failures = max_failures
        self.max_jitter = max_jitter
        self.max_backoff = max_backoff
        self.exp_base = exp_base
        self.exp_factor = exp_factor

        self.reset()

    def reset(self) -> None:
        self.failures = 0
        self.first_error = None
        self.last_error = None

    @property
    def backoff(self) -> float:
        if self.failures == 0:
            return 0
        jitter = random.uniform(0, self.max_jitter)
        return min(self.exp_factor * (self.exp_base**self.failures) + jitter, self.max_backoff)

    @property
    def remaining_attempts(self) -> int:
        return max(self.max_failures - self.failures, 0)

    def failure(self, exc: Exception | None) -> None:
        if self.failures == 0:
            self.first_error = exc
        self.last_error = exc
        self.failures += 1

    def success(self) -> None:
        self.failures = 0


class TimeoutableAsyncIterator[T](AsyncIterator[T]):
    def __init__(self, iterator: AsyncIterator[T], timeout: float) -> None:
        self.iterator = iterator
        self.timeout = timeout

    async def __anext__(self) -> T:
        return await asyncio.wait_for(self.iterator.__anext__(), timeout=self.timeout)


class ResilientEventSource:
    get_url: Callable[[], Awaitable[str]]
    get_ping_interval: Callable[[], Awaitable[float]]
    verify: bool | str
    retry_ctx: RetryContext

    logger: logging.Logger
    server_ping_grace_period: float

    _last_event_id: str
    _retry_ms: int

    active: asyncio.Future[None]

    def __init__(
        self,
        url: str | Callable[[], Awaitable[str]],
        ping_interval: float | Callable[[], Awaitable[float]] = 0.0,
        verify: bool | str = True,
        retry_ctx: RetryContext | None = None,
    ) -> None:
        self.get_url = self.async_return(url) if isinstance(url, str) else url
        if isinstance(ping_interval, int | float):
            ping_interval = self.async_return(ping_interval)
        self.get_ping_interval = ping_interval
        self.verify = verify
        self.retry_ctx = RetryContext() if retry_ctx is None else retry_ctx

        self.logger = logger
        self.server_ping_grace_period = 3.0

    def reset(self) -> None:
        self._last_event_id = ""
        self._retry_ms = 0
        self.retry_ctx.reset()
        self.active = asyncio.get_running_loop().create_future()

    @staticmethod
    def async_return[T](x: T) -> Callable[[], Awaitable[T]]:
        async def f() -> T:
            return x

        return f

    @staticmethod
    def decode_json(data: str) -> dict[str, Any] | None:
        try:
            r = json.loads(data)
        except json.JSONDecodeError:
            return None
        if type(r) is not dict:
            return None
        return cast(dict[str, Any], r)

    @property
    def headers(self) -> dict[str, str]:
        r = {"Accept": "text/event-stream"}
        if self._last_event_id:
            r["Last-Event-ID"] = self._last_event_id
        return r

    def failure(self, exc: Exception | None) -> None:
        self.active = asyncio.get_running_loop().create_future()
        self.retry_ctx.failure(exc)

    def success(self) -> None:
        self.retry_ctx.success()
        self.active.set_result(None)

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            if self.retry_ctx.remaining_attempts == 0:
                raise SSELoopStopped(
                    first_error=self.retry_ctx.first_error,
                    last_error=self.retry_ctx.last_error,
                )
            try:
                if self.retry_ctx.failures > 0:
                    self.logger.info(
                        f"SSE loop retry attempt {self.retry_ctx.failures} "
                        f"(backoff {self.retry_ctx.backoff:.3f}, retry_ms {self._retry_ms}, "
                        f"last error {self.retry_ctx.last_error})"
                    )
                    await asyncio.sleep(self.retry_ctx.backoff + self._retry_ms / 1000)
                url = await self.get_url()
                ping_interval = await self.get_ping_interval()

                async with (
                    httpx.AsyncClient(timeout=None, verify=self.verify) as c,
                    httpx_sse.aconnect_sse(c, "GET", url, headers=self.headers) as es,
                ):
                    check_status(es.response)
                    self.success()
                    if ping_interval > 0:
                        timeout = ping_interval + self.server_ping_grace_period
                        it = TimeoutableAsyncIterator(es.aiter_sse(), timeout=timeout)
                    else:
                        it = es.aiter_sse()
                    async for sse in it:
                        self._last_event_id = sse.id
                        self._retry_ms = sse.retry or 0
                        if sse.event == "ping":
                            self.logger.debug("got SSE ping")
                            continue
                        if sse.event != "message":
                            self.logger.warning(f"unexpected SSE event: {sse.event} ({sse.data})")
                            continue
                        if (event := self.decode_json(sse.data)) is None:
                            self.logger.warning(f"unexpected SSE message: {sse.data}")
                            continue
                        yield event
                    raise SSELoopStopped(message="SSE loop exited")
            except (SSELoopStopped, httpx.HTTPError, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) and not str(exc):
                    exc = TimeoutError("timeout")
                self.failure(exc)


@dc.dataclass(kw_only=True)
class LoginCredentials:
    user: str
    password: str

    @property
    def as_login_params(self) -> dict[str, str]:
        return {"username": self.user, "password": self.password}

    @property
    def description(self) -> str:
        return f"user {self.user}"


@dc.dataclass(kw_only=True)
class ApiKeyCredentials:
    api_key: str

    @property
    def as_login_params(self) -> dict[str, str]:
        return {"api_key": self.api_key}

    @property
    def description(self) -> str:
        return f"API key {self.api_key[:13]}..."


type Credentials = LoginCredentials | ApiKeyCredentials


class EditorAPIContext:
    credentials: Credentials
    base_url: str
    priority: Priority
    verify: bool | str
    default_timeout: float
    user_agent: str

    token: str | None
    logger: logging.Logger
    credits: int | None = None

    _client: httpx.AsyncClient | None
    _client_ctx_depth: int
    _sse_futures: Futures[StateID, dict[str, Any]]
    _sse_source: ResilientEventSource
    _sse_task: asyncio.Task[None] | None
    _ping_interval: float

    def __init__(
        self,
        credentials: str | None = None,
        api_key: str | None = None,
        user: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
        priority: Priority = "standard",
        verify: bool | str = True,
        default_timeout: float = 60.0,
        user_agent: str | None = None,
    ) -> None:
        self.base_url = base_url or "https://api.finegrain.ai/editor"
        self.priority = priority
        self.verify = verify
        self.default_timeout = default_timeout

        if credentials is not None:
            if (m := API_KEY_PATTERN.match(credentials)) is not None:
                self.credentials = ApiKeyCredentials(api_key=m[0])
            elif (m := EMAIL_PWD_PATTERN.match(credentials)) is not None:
                self.credentials = LoginCredentials(user=m["email"], password=m["pwd"])
            else:
                raise ValueError("invalid credentials")
        elif api_key is not None:
            self.credentials = ApiKeyCredentials(api_key=api_key)
        elif user is not None and password is not None:
            self.credentials = LoginCredentials(user=user, password=password)
        else:
            raise ValueError("either `api_key` or `user` and `password` must be provided")

        client_ua = f"finegrain-python/{VERSION}"
        if user_agent is None:
            self.user_agent = client_ua
        else:
            self.user_agent = f"{user_agent} ({client_ua})"

        self.logger = logger
        self._sse_source = ResilientEventSource(
            url=self.get_sub_url,
            ping_interval=self.get_ping_interval,
            verify=self.verify,
        )
        self.reset()

    def reset(self) -> None:
        self.token = None
        self._client = None
        self._client_ctx_depth = 0
        self._sse_futures = Futures()
        self._sse_task = None
        self._ping_interval = 0.0
        try:
            self._sse_source.reset()
        except RuntimeError:  # outside asyncio
            pass

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._client:
            assert self._client_ctx_depth > 0
            self._client_ctx_depth += 1
            return self._client
        assert self._client_ctx_depth == 0
        self._client = httpx.AsyncClient(verify=self.verify, headers={"User-Agent": self.user_agent})
        self._client_ctx_depth = 1
        return self._client

    async def __aexit__(self, *args: Any) -> None:
        if (not self._client) or self._client_ctx_depth <= 0:
            raise RuntimeError("unbalanced __aexit__")
        self._client_ctx_depth -= 1
        if self._client_ctx_depth == 0:
            await self._client.__aexit__(*args)
            self._client = None

    @property
    def auth_headers(self) -> dict[str, str]:
        assert self.token
        return {"Authorization": f"Bearer {self.token}"}

    async def login(self) -> None:
        async with self as client:
            response = await client.post(
                f"{self.base_url}/auth/login",
                json=self.credentials.as_login_params,
            )
        check_status(response)
        self.logger.debug(f"logged in as {self.credentials.description}")
        r = response.json()
        self.credits = r["user"]["credits"]
        self.token = r["token"]

    async def request(
        self,
        method: Literal["GET", "POST"],
        url: str,
        files: RequestFiles | None = None,
        data: RequestData | None = None,
        params: QueryParamTypes | None = None,
        json: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        async def _q() -> httpx.Response:
            return await client.request(
                method,
                f"{self.base_url}/{url}",
                headers=dict(headers or {}) | self.auth_headers,
                files=files,
                data=data,
                params=params,
                json=json,
            )

        async with self as client:
            r = await _q()
            if r.status_code == 401:
                self.logger.debug("renewing token")
                await self.login()
                r = await _q()

        if raise_for_status:
            check_status(r)
        return r

    async def get_sub_url(self) -> str:
        response = await self.request("POST", "sub-auth")
        jdata = response.json()
        sub_token = jdata["token"]
        self._ping_interval = float(jdata.get("ping_interval", 0.0))
        return f"{self.base_url}/sub/{sub_token}"

    async def get_ping_interval(self) -> float:
        return self._ping_interval

    async def _sse_loop(self) -> None:
        async for event in self._sse_source:
            if "state" not in event:
                self.logger.warning(f"unexpected SSE message: {event}")
                continue
            self.logger.debug(f"got message: {event}")
            self._sse_futures[event["state"]].set_result(event)
            if "credits_left" in event:
                self.credits = event["credits_left"]

    async def sse_start(self) -> None:
        assert self._sse_task is None
        self._sse_source.reset()
        self._sse_task = asyncio.create_task(self._sse_loop())
        await self._sse_source.active

    async def sse_stop(self) -> None:
        assert self._sse_task
        self._sse_task.cancel()
        exc = await asyncio.gather(self._sse_task, return_exceptions=True)
        assert len(exc) == 1 and isinstance(exc[0], asyncio.CancelledError)
        self._sse_task = None

    async def sse_await(self, state_id: StateID, timeout: float | None = None) -> bool:
        assert self._sse_task
        future = self._sse_futures[state_id]
        timeout = timeout or self.default_timeout

        sse_task = self._sse_task
        done, _ = await asyncio.wait(
            {future, self._sse_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sse_task in done:
            exception = sse_task.exception()
            raise SSELoopStopped(f"SSE loop stopped while waiting for state {state_id}") from exception
        if not done:
            r = await self.request("GET", f"state/meta/{state_id}", raise_for_status=False)
            if r.is_success:
                status = r.json()["status"]
                self.logger.warning(f"got timeout for state {state_id}, found metadata with status {status}")
                return status == "ok"
            elif r.status_code != 404:
                raise TimeoutError(f"state {state_id} timed out after {timeout}")
            else:
                raise RuntimeError(f"getting state {state_id} after timeout {timeout} returned {r.status_code}")

        assert done == {future}

        event = future.result()
        del self._sse_futures[state_id]
        return event["status"] == "ok"

    async def get_meta(self, state_id: StateID) -> dict[str, Any]:
        response = await self.request("GET", f"state/meta/{state_id}")
        return response.json()

    async def get_image(
        self,
        state_id: StateID,
        image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"] = "AUTO",
        resolution: Literal["FULL", "DISPLAY"] = "FULL",
    ) -> bytes:
        params = {"format": image_format, "resolution": resolution}
        response = await self.request("GET", f"state/image/{state_id}", params=params)
        return response.content

    async def _run_one[Tin, Tout](
        self,
        co: Callable[["EditorAPIContext", Tin], Awaitable[Tout]],
        params: Tin,
    ) -> Tout:
        # This wraps the coroutine in the SSE loop.
        # This is mostly useful if you use synchronous Python,
        # otherwise you can call the functions directly.
        if not self.token:
            await self.login()
        await self.sse_start()
        try:
            r = await co(self, params)
            return r
        finally:
            await self.sse_stop()

    def run_one_sync[Tin, Tout](
        self,
        co: Callable[["EditorAPIContext", Tin], Awaitable[Tout]],
        params: Tin,
    ) -> Tout:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_one(co, params))

    async def call_skill(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[StateID, bool]:
        params = {"priority": self.priority} | (params or {})
        response = await self.request("POST", f"skills/{url}", json=params)
        state_id: StateID = response.json()["state"]
        status = await self.sse_await(state_id, timeout=timeout)
        return state_id, status

    async def ensure_skill(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> StateID:
        st, ok = await self.call_skill(url, params, timeout=timeout)
        if ok:
            return st
        meta = await self.get_meta(st)
        raise RuntimeError(f"skill {url} failed with {st}: {meta}")

    @property
    def call_async(self) -> "EditorApiAsyncClient":
        return EditorApiAsyncClient(self)


## High-level interface ##

CreateStateErrorCode = Literal["file_too_large", "download_error", "invalid_image"]
Trinary = Literal["yes", "no", "unknown"]
Size2D = tuple[int, int]
BoundingBox = tuple[int, int, int, int]
Mode = Literal["express", "standard", "premium"]


def _size2d(v: Any) -> Size2D:
    assert isinstance(v, list)
    v = cast(list[Any], v)
    assert all(isinstance(x, int) for x in v)
    r = cast(tuple[int, ...], tuple(v))
    assert len(r) == 2
    return r


def _bbox(v: Any) -> BoundingBox:
    assert isinstance(v, list)
    v = cast(list[Any], v)
    assert all(isinstance(x, int) for x in v)
    r = cast(tuple[int, ...], tuple(v))
    assert len(r) == 4
    return r


def _color(v: Any) -> tuple[int, int, int] | tuple[int, int, int, int]:
    assert isinstance(v, list)
    v = cast(list[Any], v)
    assert all(isinstance(x, int) for x in v)
    if len(v) == 3:
        return cast(tuple[int, int, int], tuple(v))
    elif len(v) == 4:
        return cast(tuple[int, int, int, int], tuple(v))
    else:
        raise ValueError(f"unexpected color: {v}")


@dc.dataclass(kw_only=True)
class MetaResult:
    state_id: StateID
    meta: dict[str, Any]


class OKResult(MetaResult):
    @property
    def input_states(self) -> list[StateID]:
        v = self.meta.get("input_states", [])
        assert isinstance(v, list)
        v = cast(list[Any], v)
        assert all(isinstance(x, str) for x in v)
        return cast(list[StateID], v)

    @property
    def image_size(self) -> Size2D:
        return _size2d(self.meta["image_size"])

    @property
    def credit_cost(self) -> int:
        v = self.meta["credit_cost"]
        assert isinstance(v, int)
        return v


@dc.dataclass(kw_only=True)
class OKResultWithImage(OKResult):
    image: bytes


class ErrorResult(MetaResult):
    @property
    def error(self) -> str:
        v = self.meta["error"]
        assert isinstance(v, str)
        return v


class CreateStateResult(OKResult):
    @property
    def original_mimetype(self) -> str:
        v = self.meta["original_mimetype"]
        assert isinstance(v, str)
        return v


class CreateStateError(ErrorResult):
    @property
    def error_code(self) -> CreateStateErrorCode:
        v = self.meta["error_code"]
        assert v in get_args(CreateStateErrorCode)
        return v


class InferIsProductResult(OKResult):
    @property
    def is_product(self) -> Trinary:
        v = self.meta["is_product"]
        assert v in get_args(Trinary)
        return v


class InferProductNameResult(OKResult):
    @property
    def is_product(self) -> str:
        v = self.meta["product_name"]
        assert isinstance(v, str)
        return v


class InferMainSubjectResult(OKResult):
    @property
    def main_subject(self) -> str:
        v = self.meta["main_subject"]
        assert isinstance(v, str)
        return v


class InferCommercialDescriptionResult(OKResult):
    @property
    def commercial_description_en(self) -> str:
        v = self.meta["commercial_description_en"]
        assert isinstance(v, str)
        return v


class InferBoundingBoxResult(OKResult):
    @property
    def bbox(self) -> BoundingBox:
        return _bbox(self.meta["bbox"])


class SegmentResult(OKResult):
    pass


class SegmentResultWithImage(OKResultWithImage, SegmentResult):
    @property
    def mask(self) -> bytes:
        return self.image


class OKResultWithUsedSeeds(OKResult):
    @property
    def used_seeds(self) -> list[int]:
        v = self.meta.get("used_seeds", [])
        assert isinstance(v, list)
        v = cast(list[Any], v)
        assert all(isinstance(x, int) for x in v)
        return cast(list[int], v)


class EraseResult(OKResultWithUsedSeeds):
    pass


class EraseResultWithImage(OKResultWithImage, EraseResult):
    pass


class BlendResult(OKResultWithUsedSeeds):
    @property
    def input_bbox(self) -> BoundingBox:
        return _bbox(self.meta["input_bbox"])

    @property
    def blended_bbox(self) -> BoundingBox:
        return _bbox(self.meta["blended_bbox"])

    @property
    def crop_bbox(self) -> BoundingBox | None:
        if "crop_bbox" not in self.meta:
            return None
        return _bbox(self.meta["crop_bbox"])


class BlendResultWithImage(OKResultWithImage, BlendResult):
    pass


class UpscaleResult(OKResultWithUsedSeeds):
    pass


class UpscaleResultWithImage(OKResultWithImage, UpscaleResult):
    pass


class ShadowResult(OKResultWithUsedSeeds):
    @property
    def input_bbox(self) -> BoundingBox | None:
        if "input_bbox" not in self.meta:
            return None
        return _bbox(self.meta["input_bbox"])

    @property
    def output_bbox(self) -> BoundingBox:
        return _bbox(self.meta["output_bbox"])

    @property
    def crop_bbox(self) -> BoundingBox | None:
        if "crop_bbox" not in self.meta:
            return None
        return _bbox(self.meta["crop_bbox"])


class ShadowResultWithImage(OKResultWithImage, ShadowResult):
    pass


class RecolorResult(OKResult):
    @property
    def color(self) -> tuple[int, int, int] | tuple[int, int, int, int]:
        return _color(self.meta["color"])


class RecolorResultWithImage(OKResultWithImage, RecolorResult):
    pass


class CutoutResult(OKResult):
    @property
    def mask_bbox(self) -> BoundingBox:
        return _bbox(self.meta["mask_bbox"])


class CutoutResultWithImage(OKResultWithImage, CutoutResult):
    pass


class CropResult(OKResult):
    @property
    def crop_bbox(self) -> BoundingBox:
        return _bbox(self.meta["crop_bbox"])


class CropResultWithImage(OKResultWithImage, CropResult):
    pass


class MergeMasksResult(OKResult):
    pass


class MergeMasksResultWithImage(OKResultWithImage, MergeMasksResult):
    pass


class SetBackgroundColorResult(OKResult):
    @property
    def background(self) -> tuple[int, int, int] | tuple[int, int, int, int]:
        return _color(self.meta["background"])


class SetBackgroundColorResultWithImage(OKResultWithImage, SetBackgroundColorResult):
    pass


@dc.dataclass(kw_only=True)
class MergeCutoutsEntry:
    state_id: StateID
    bbox: BoundingBox
    flip: bool = False
    rotation_angle: float = 0.0

    @property
    def as_options(self) -> dict[str, Any]:
        r: dict[str, Any] = {"bbox": list(self.bbox)}
        if self.flip:
            r["flip"] = True
        if self.rotation_angle:
            r["rotation_angle"] = self.rotation_angle
        return r


class MergeCutoutsResult(OKResult):
    pass


class MergeCutoutsResultWithImage(OKResultWithImage, MergeCutoutsResult):
    pass


@dc.dataclass(kw_only=True)
class ImageOutParams:
    image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"] = "AUTO"
    resolution: Literal["FULL", "DISPLAY"] = "FULL"


class EditorApiAsyncClient:
    def __init__(self, ctx: EditorAPIContext) -> None:
        self.ctx = ctx

    async def upload_image(self, file: BinaryIO | bytes) -> StateID:
        response = await self.ctx.request("POST", "state/upload", files={"file": file})
        return response.json()["state"]

    async def _create_state(
        self,
        file: BinaryIO | bytes | None,
        file_url: str | None = None,
        meta: dict[str, Any] | None = None,
        timeout: float | None = 30.0,
    ) -> tuple[StateID, bool]:
        if (file is not None) and (file_url is not None):
            raise ValueError("cannot specify both file and file_url")
        files = None if file is None else {"file": file}
        data: dict[str, str] = {}
        if file_url is not None:
            data["file_url"] = file_url
        if meta is not None:
            data["meta"] = json.dumps(meta)
        response = await self.ctx.request("POST", "state/create", files=files, data=data)
        state_id: StateID = response.json()["state"]
        status = await self.ctx.sse_await(state_id, timeout=timeout)
        return state_id, status

    async def _response[Tok: OKResult, Tko: ErrorResult](
        self,
        st: StateID,
        ok: bool,
        t_ok: type[Tok] = OKResult,
        t_ko: type[Tko] = ErrorResult,
    ) -> Tok | Tko:
        meta = await self.ctx.get_meta(st)
        if ok:
            assert meta["status"] == "ok"
            return t_ok(state_id=st, meta=meta)
        else:
            assert meta["status"] == "ko"
            return t_ko(state_id=st, meta=meta)

    async def _response_with_image[Tok: OKResultWithImage, Tko: ErrorResult](
        self,
        st: StateID,
        ok: bool,
        t_ok: type[Tok] = OKResultWithImage,
        t_ko: type[Tko] = ErrorResult,
        params: ImageOutParams | None = None,
    ) -> Tok | Tko:
        if ok:
            if params is None:
                params = ImageOutParams()
            async with asyncio.TaskGroup() as tg:
                meta_f = tg.create_task(self.ctx.get_meta(st))
                image_f = tg.create_task(self.ctx.get_image(st, params.image_format, params.resolution))
            meta = meta_f.result()
            image = image_f.result()
            assert meta["status"] == "ok"
            return t_ok(state_id=st, meta=meta, image=image)
        else:
            meta = await self.ctx.get_meta(st)
            assert meta["status"] == "ko"
            return t_ko(state_id=st, meta=meta)

    async def create_state(
        self,
        file: BinaryIO | bytes | None = None,
        file_url: str | None = None,
        meta: dict[str, Any] | None = None,
        timeout: float | None = 30.0,
    ) -> CreateStateResult | CreateStateError:
        st, ok = await self._create_state(file, file_url, meta, timeout)
        return await self._response(st, ok, CreateStateResult, CreateStateError)

    async def infer_is_product(
        self,
        state_id: StateID,
        timeout: float | None = None,
    ) -> InferIsProductResult | ErrorResult:
        st, ok = await self.ctx.call_skill(f"infer-is-product/{state_id}", timeout=timeout)
        return await self._response(st, ok, InferIsProductResult)

    async def infer_product_name(
        self,
        state_id: StateID,
        timeout: float | None = None,
    ) -> InferProductNameResult | ErrorResult:
        st, ok = await self.ctx.call_skill(f"infer-product-name/{state_id}", timeout=timeout)
        return await self._response(st, ok, InferProductNameResult)

    async def infer_main_subject(
        self,
        state_id: StateID,
        timeout: float | None = None,
    ) -> InferMainSubjectResult | ErrorResult:
        st, ok = await self.ctx.call_skill(f"infer-main-subject/{state_id}", timeout=timeout)
        return await self._response(st, ok, InferMainSubjectResult)

    async def infer_commercial_description(
        self,
        state_id: StateID,
        product_name: str | None = None,
        timeout: float | None = None,
    ) -> InferCommercialDescriptionResult | ErrorResult:
        params: dict[str, Any] = {}
        if product_name is not None:
            params["product_name"] = product_name
        st, ok = await self.ctx.call_skill(
            f"infer-commercial-description/{state_id}",
            params,
            timeout=timeout,
        )
        return await self._response(st, ok, InferCommercialDescriptionResult)

    async def infer_bbox(
        self,
        state_id: StateID,
        product_name: str | None = None,
        timeout: float | None = None,
    ) -> InferBoundingBoxResult | ErrorResult:
        params: dict[str, Any] = {}
        if product_name is not None:
            params["product_name"] = product_name
        st, ok = await self.ctx.call_skill(
            f"infer-bbox/{state_id}",
            params,
            timeout=timeout,
        )
        return await self._response(st, ok, InferBoundingBoxResult)

    async def segment(
        self,
        state_id: StateID,
        bbox: BoundingBox | None = None,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> SegmentResult | ErrorResult:
        params: dict[str, Any] = {}
        if bbox is not None:
            params["bbox"] = list(bbox)
        st, ok = await self.ctx.call_skill(f"segment/{state_id}", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, SegmentResultWithImage, params=image_params)
        return await self._response(st, ok, SegmentResult)

    async def erase(
        self,
        image_state_id: StateID,
        mask_state_id: StateID,
        seed: int | None = None,
        mode: Mode = "standard",
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> EraseResult | ErrorResult:
        params: dict[str, Any] = {"mode": mode}
        if seed is not None:
            params["seed"] = seed
        st, ok = await self.ctx.call_skill(
            f"erase/{image_state_id}/{mask_state_id}",
            params,
            timeout=timeout,
        )
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, EraseResultWithImage, params=image_params)
        return await self._response(st, ok, EraseResult)

    async def blend(
        self,
        image_state_id: StateID,
        mask_state_id: StateID,
        bbox: BoundingBox | None = None,
        flip: bool = False,
        rotation_angle: float = 0.0,
        seed: int | None = None,
        mode: Mode = "standard",
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> BlendResult | ErrorResult:
        params: dict[str, Any] = {
            "mode": mode,
            "flip": flip,
            "rotation_angle": rotation_angle,
        }
        if bbox is not None:
            params["bbox"] = list(bbox)
        if seed is not None:
            params["seed"] = seed
        st, ok = await self.ctx.call_skill(
            f"blend/{image_state_id}/{mask_state_id}",
            params,
            timeout=timeout,
        )
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, BlendResultWithImage, params=image_params)
        return await self._response(st, ok, BlendResult)

    async def upscale(
        self,
        state_id: StateID,
        preprocess: bool = True,
        scale_factor: Literal[1, 2, 4] = 2,
        resemblance: float | None = None,
        decay: float | None = None,
        creativity: float | None = None,
        seed: int | None = None,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> UpscaleResult | ErrorResult:
        params: dict[str, Any] = {"preprocess": preprocess, "scale_factor": scale_factor}
        if resemblance is not None:
            params["resemblance"] = resemblance
        if decay is not None:
            params["decay"] = decay
        if creativity is not None:
            params["creativity"] = creativity
        if seed is not None:
            params["seed"] = seed
        st, ok = await self.ctx.call_skill(f"upscale/{state_id}", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, UpscaleResultWithImage, params=image_params)
        return await self._response(st, ok, UpscaleResult)

    async def shadow(
        self,
        state_id: StateID,
        resolution: Size2D | None = None,
        bbox: BoundingBox | None = None,
        background: str | None = None,
        seed: int | None = None,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> ShadowResult | ErrorResult:
        params: dict[str, Any] = {}
        if resolution is not None:
            params["resolution"] = list(resolution)
        if bbox is not None:
            params["bbox"] = list(bbox)
        if background is not None:
            params["background"] = background
        if seed is not None:
            params["seed"] = seed
        st, ok = await self.ctx.call_skill(f"shadow/{state_id}", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, ShadowResultWithImage, params=image_params)
        return await self._response(st, ok, ShadowResult)

    async def recolor(
        self,
        image_state_id: StateID,
        mask_state_id: StateID,
        color: str,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> RecolorResult | ErrorResult:
        params: dict[str, Any] = {"color": color}
        st, ok = await self.ctx.call_skill(
            f"recolor/{image_state_id}/{mask_state_id}",
            params,
            timeout=timeout,
        )
        if with_image:
            return await self._response_with_image(st, ok, RecolorResultWithImage)
        return await self._response(st, ok, RecolorResult)

    async def cutout(
        self,
        image_state_id: StateID,
        mask_state_id: StateID,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> CutoutResult | ErrorResult:
        st, ok = await self.ctx.call_skill(f"cutout/{image_state_id}/{mask_state_id}", timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, CutoutResultWithImage, params=image_params)
        return await self._response(st, ok, CutoutResult)

    async def crop(
        self,
        state_id: StateID,
        bbox: BoundingBox | None = None,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> CropResult | ErrorResult:
        params: dict[str, Any] = {}
        if bbox is not None:
            params["bbox"] = list(bbox)
        st, ok = await self.ctx.call_skill(f"crop/{state_id}", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, CropResultWithImage, params=image_params)
        return await self._response(st, ok, CropResult)

    async def merge_masks(
        self,
        state_ids: list[StateID],
        operation: Literal["union", "difference"] = "union",
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> MergeMasksResult | ErrorResult:
        params: dict[str, Any] = {"operation": operation, "states": state_ids}
        st, ok = await self.ctx.call_skill("merge-masks", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, MergeMasksResultWithImage, params=image_params)
        return await self._response(st, ok, MergeMasksResult)

    async def merge_cutouts(
        self,
        resolution: Size2D,
        cutouts: list[MergeCutoutsEntry],
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> MergeCutoutsResult | ErrorResult:
        state_ids: list[StateID] = [e.state_id for e in cutouts]
        options: list[dict[str, Any]] = [e.as_options for e in cutouts]
        params: dict[str, Any] = {"resolution": resolution, "states": state_ids, "options": options}
        st, ok = await self.ctx.call_skill("merge-cutouts", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, MergeCutoutsResultWithImage, params=image_params)
        return await self._response(st, ok, MergeCutoutsResult)

    async def set_background_color(
        self,
        state_id: StateID,
        background: str,
        with_image: bool | ImageOutParams = False,
        timeout: float | None = None,
    ) -> SetBackgroundColorResult | ErrorResult:
        params: dict[str, Any] = {"background": background}
        st, ok = await self.ctx.call_skill(f"set-background-color/{state_id}", params, timeout=timeout)
        if with_image:
            image_params = None if isinstance(with_image, bool) else with_image
            return await self._response_with_image(st, ok, SetBackgroundColorResultWithImage, params=image_params)
        return await self._response(st, ok, SetBackgroundColorResult)

    async def download_image(self, st: StateID) -> Image.Image:
        response = await self.ctx.get_image(st)
        return Image.open(io.BytesIO(response))


@cache
def _get_ctx() -> EditorAPIContext:
    config_path = Path(__file__).parent.parent / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found at {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    credentials = config.get("finegrain", "credentials")
    priority = config.get("finegrain", "priority")
    timeout = config.getfloat("finegrain", "timeout")

    assert priority in get_args(Priority), f"invalid priority {priority}, must be one of {get_args(Priority)}"
    priority = cast(Priority, priority)

    ctx = EditorAPIContext(
        credentials=credentials,
        priority=priority,
        default_timeout=timeout,
        user_agent="comfyui-finegrain/1.4.0",
    )

    return ctx
