# comfyui-finegrain

[Finegrain API](https://api.finegrain.ai/doc/) ComfyUI custom nodes.

## Account creation (Prerequisite)

To use these custom nodes, you need Finegrain API credentials:

1. Sign up for an account at https://editor.finegrain.ai/signup:

![finegrain login](assets/finegrain_login.webp)

2. Modify [`config.ini`](config.ini) with your Finegrain API credentials:

```ini
[finegrain]
credentials = myemail@email.com:my_password
priority = low
timeout = 60
```

> [!Note]
> `credentials` can either be `email:password` or an API key.

## Installation

Installing the nodes is pretty straight forward, check out our [Discord server](https://discord.gg/a4w4jXJ6) if you need help.

### Requirements

1. Ensure you have [Python](https://www.python.org/) 3.12 (or later) installed.

2. Ensure you have [Git](https://git-scm.com/) installed.

3. Ensure you have [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed.

### ComfyUI Manager installation (recommended)

1. Ensure you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#comfyui-manager) installed.

2. In ComfyUI, open the manager by clicking the `Manager` button in the top right corner.

3. Click `Custom Nodes Manager` in the menu.

4. Search for `comfyui-finegrain` in the search bar and click the `Install` button.

Alternatively, if you load one of the [workflows](#workflow-examples) below, you should be able to install the nodes directly
by clicking the `Install Missing Custom Nodes` button in the Manager's menu.

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
.\python_embeded\Scripts\pip.exe install -r .\ComfyUI\custom_nodes\comfyui-finegrain\requirements.txt
```

### Manual installation

The nodes are published at https://registry.comfy.org/publishers/finegrain/nodes/comfyui-finegrain.

1. Clone the repository:

```bash
cd custom_nodes
git clone https://github.com/finegrain-ai/comfyui-finegrain.git
```

2. Install the nodes' requirements:

On Linux:
```bash
# ensure you activated the python virtual environment used by ComfyUI
pip install -r custom_nodes/comfyui-finegrain/requirements.txt
```

On Windows:
```shell
.\python_embeded\Scripts\pip.exe install -r .\ComfyUI\custom_nodes\comfyui-finegrain\requirements.txt
```

## Workflow examples

> [!Note]
> All the below workflow examples were made using comfyui-finegrain v2.0.0. <br>
> To import them into ComfyUI, drag and drop the .png file into the ComfyUI window.

All our workflows are also available on OpenArt: https://openart.ai/workflows/profile/finegrain

### Prompt to erase

Instantly remove any object, along with its shadows and reflections, just by naming it.

![Prompt to erase workflow](assets/workflows/eraser.png?raw=true)

### Prompt to cutout

Instantly isolate any object in a photo into a perfect cutout, just by naming it.

![Prompt to cutout workflow](assets/workflows/cutout.png?raw=true)

### Prompt to recolor

Instantly change the color of any object in a photo, even through occlusions, just by naming it.

![Prompt to recolor workflow](assets/workflows/recolor.png?raw=true)

### Swap

Replace any object in a photo with another, recreating shadows and reflections so naturally it looks like the new object was always there — perfectly preserved in every detail.

![Swap workflow](assets/workflows/swap.png?raw=true)

### Blend

Seamlessly integrate any object into a scene, recreating shadows and reflections for a result so natural it looks like it was always there — perfectly preserved in every detail.

![Blend workflow](assets/workflows/blender.png?raw=true)

### Generate packshot

Generate Packshot – Transform any mundane photo into a stunning white-background image with a perfectly natural shadow.

![Generate packshot workflow](assets/workflows/packshot.png?raw=true)

### Remove background

Remove Background – Our pixel-perfect, high-resolution take on a classic, effortlessly extracting the main object from its background.

![Remove background workflow](assets/workflows/removebg.png?raw=true)
