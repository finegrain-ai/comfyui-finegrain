# comfyui-finegrain

[Finegrain API](https://api.finegrain.ai/doc/) ComfyUI custom nodes.

## Account creation (Prerequisite)

To use these custom nodes, you need Finegrain API credentials:

1. Sign up for an account at https://editor.finegrain.ai/signup.

2. Use your **username** (email) and **password** in the `Finegrain API` node.

> [!Note]
> Behind the scenes, authentication is handled by the [Finegrain Python client](https://github.com/finegrain-ai/finegrain-python/),
> so you don't need to manage [API tokens](https://api.finegrain.ai/doc/authentication/) manually.

## Installation

Installing the nodes is pretty straight forward, check out our [Discord server](https://discord.gg/a4w4jXJ6) if you need help.

### Requirements

1. Ensure you have **Python 3.12** (or later) installed.

2. Ensure you have Git installed.

### ComfyUI Manager

1. Ensure you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#comfyui-manager) installed.

2. In ComfyUI, open the manager by clicking the `Manager` button in the top right corner.

3. Click `Custom Nodes Manager` in the menu.

4. Search for `comfyui-finegrain` in the search bar and click the `Install` button.

### Comfy Registry installation

The nodes are published at https://registry.comfy.org/publishers/finegrain/nodes/comfyui-finegrain.

1. Ensure you have [Comfy CLI](https://docs.comfy.org/comfy-cli/getting-started) installed.

2. Install the custom nodes using Comfy CLI:

```bash
comfy node registry-install comfyui-finegrain
```

The above command should automatically install the nodes' requirements.
If it somehow doesn't, you can manually install them with:

On Linux:
```bash
# ensure you activated the python virtual environment used by ComfyUI
pip install -r custom_nodes/comfyui-finegrain/requirements.txt
```

On Windows:
```shell
.\python_embeded\Scripts\pip.exe install hatchling
.\python_embeded\Scripts\pip.exe install -r .\ComfyUI\custom_nodes\comfyui-finegrain\requirements.txt
```

## Workflow examples

> [!Note]
> All the below workflow examples were made using comfyui-finegrain v1.0.0.

All our workflows are also available on OpenArt: https://openart.ai/workflows/profile/finegrain

### Prompt to erase

Instantly remove any object, along with its shadows and reflections, just by naming it.

![Prompt to erase workflow](assets/workflows/erase.webp?raw=true)

[Download the Prompt to erase workflow](assets/workflows/erase.json)

### Prompt to cutout

Instantly isolate any object in a photo into a perfect cutout, just by naming it.

![Prompt to cutout workflow](assets/workflows/cutout.webp?raw=true)

[Download the Prompt to cutout workflow](assets/workflows/cutout.json)

### Prompt to recolor

Instantly change the color of any object in a photo, even through occlusions, just by naming it.

![Prompt to recolor workflow](assets/workflows/recolor.webp?raw=true)

[Download the Prompt to recolor workflow](assets/workflows/recolor.json)

### Swap

Replace any object in a photo with another, recreating shadows and reflections so naturally it looks like the new object was always there — perfectly preserved in every detail.

![Swap workflow](assets/workflows/swap.webp?raw=true)

[Download the Swap workflow](assets/workflows/swap.json)

### Blend

Seamlessly integrate any object into a scene, recreating shadows and reflections for a result so natural it looks like it was always there — perfectly preserved in every detail.

![Blend workflow](assets/workflows/blend.webp?raw=true)

[Download the Blend workflow](assets/workflows/blend.json)

### Generate packshot

Generate Packshot – Transform any mundane photo into a stunning white-background image with a perfectly natural shadow.

![Generate packshot workflow](assets/workflows/packshot.webp?raw=true)

[Download the Generate packshot workflow](assets/workflows/packshot.json)

### Remove background

Remove Background – Our pixel-perfect, high-resolution take on a classic, effortlessly extracting the main object from its background.

![Remove background workflow](assets/workflows/removebg.webp?raw=true)

[Download the Remove background workflow](assets/workflows/removebg.json)
