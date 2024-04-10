from random import choice, choices, randint
import time
from io import BytesIO
from pathlib import Path
from typing import Literal

import PIL
import PIL.Image
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.fields import InputField, MetadataField, OutputField
from invokeai.app.invocations.image import ImageOutput
from novelai_api.ImagePreset import (
    ImageModel,
    ImagePreset,
    ImageSampler,
    UCPreset,
)
from novelai_api.NovelAI_API import NovelAIAPI as Api
from invokeai.app.invocations.primitives import StringCollectionOutput, StringOutput
from numpy import choose

from .utils import run_coro

I = InputField
O = OutputField


@invocation("file_readlines", title="File readlines", category="Tickt")
class FileReadLineInvocation(BaseInvocation):
    """Load file content by lines (will ignore blank line)"""

    file_path: str = I()

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        with open(self.file_path, "r") as f:
            return StringCollectionOutput(
                collection=list(filter(lambda x: x, [i.strip() for i in f.readlines()]))
            )


@invocation("strlis_random_choice", title="Strlis random choice", category="Tickt")
class ChooseSomeStrInvocation(BaseInvocation):
    """0~1 as percentage, others as number"""

    input: list[str] = I()
    min: float = I(0, ge=0)
    max: float = I(0.999)
    order: bool = I(False)
    cached_file_path: str = I("")

    def choose(self):
        self.input = list(filter(lambda x: x, self.input))
        l = len(self.input)
        _min = self.min if self.min >= 1 else l * self.min
        _max = self.max if self.max >= 1 else l * self.max
        _min = int(_min)
        _max = int(_max)

        res = []
        ll = list(range(l))
        for i in range(min(l, randint(_min, _max))):
            p = choice(ll)
            ll.remove(p)
            res.append(self.input[p])
        return res

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        s = set()
        if self.cached_file_path != "":
            with open(self.cached_file_path, "r") as f:
                for i in f:
                    s.add(i.strip())
        while True:
            p = self.choose()
            if not self.order:
                p = sorted(p)
            key = "".join(p)
            if key not in s:
                s.add(key)
                if self.cached_file_path:
                    with open(self.cached_file_path, "w") as f:
                        f.writelines([i + "\n" for i in s])
                return StringCollectionOutput(collection=p)


@invocation("str_joins", title="String joins", category="Tickt")
class StringJoinsInvocation(BaseInvocation):
    input: list[str] = I()
    symbol: str = I("")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=f"{self.symbol}".join(self.input) + self.symbol)


@invocation("nai_txt_to_img", title="Nai Txt2img", category="Tickt")
class NaiTxtToImgInvocation(BaseInvocation):
    repeat: int = I(1, ge=1)
    token: str = I()
    limit_free: bool = I(True)
    height: int = I(default=832, ge=64, le=1728)
    width: int = I(default=1216, ge=64, le=1728)
    prompt: str = I()
    uc: str = I()
    seed: int = I()
    step: int = I(28, ge=1, le=50)
    uc_preset: Literal["None", "Light", "Heavy", "Human Focus"] = I("Heavy")
    sampler: Literal[
        "k_euler",
        "k_euler_ancestral",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_2m",
        "k_dpmpp_sde",
        "ddim",
    ] = I("k_euler_ancestral")
    smea: Literal["none", "SMEA", "SMEA+DYN"] = I("SMEA+DYN")
    prompt_guidance: float = I(6, ge=0, le=10)
    prompt_guidance_rescale: float = I(0, ge=0, le=1)
    uc_strength: float = I(1, ge=0, le=10)
    noise: Literal["native", "exponential", "polyexponential"] = I("native")
    save: bool = I(False)
    save_path: str = I()

    async def hook(self) -> tuple[str, bytes]:

        # login
        client = Api()
        await client.high_level.login_with_token(self.token)

        # set config
        preset = ImagePreset.from_v3_config()
        preset.resolution = (self.height, self.width)
        preset.steps = self.step
        preset.seed = self.seed
        preset.smea_dyn = self.smea == "SMEA+DYN"
        preset.smea = (self.smea == "SMEA") | preset.smea_dyn
        preset.scale = self.prompt_guidance
        preset.uncond_scale = self.uc_strength
        preset.cfg_rescale = self.prompt_guidance_rescale
        preset.noise_schedule = self.noise

        p = UCPreset
        s = self.uc_preset
        preset.uc = self.uc
        preset.uc_preset = (
            p.Preset_None
            if s == "None"
            else (
                p.Preset_Light
                if s == "Light"
                else p.Preset_Heavy if s == "Heavy" else p.Preset_Bad_Anatomy
            )
        )

        p = ImageSampler
        s = self.sampler
        preset.sampler = (
            p.k_euler
            if s == "k_euler"
            else (
                p.k_euler_ancestral
                if s == "k_euler_ancestral"
                else (
                    p.k_dpm_2
                    if s == "k_dpmpp_2m"
                    else (
                        p.k_dpm_2_ancestral
                        if s == "k_dpmpp_2s_ancestral"
                        else p.k_dpmpp_sde if s == "k_dpmpp_sde" else p.ddim
                    )
                )
            )
        )

        p = []
        async for i in client.high_level.generate_image(
            self.prompt, ImageModel.Anime_v3, preset
        ):
            p.append(i)
        return p[0]

    def invoke(self, c: InvocationContext) -> ImageOutput:
        logger = c.logger
        for i in range(self.repeat):
            try:
                logger.info(f"Running loop {i+1}")
                res = run_coro(self.hook())
                if self.save:
                    path = Path(self.save_path) / f"{int(time.time())}.png"
                    with open(path, "wb") as f:
                        f.write(res[1])
            except Exception as e:
                logger.error(f"Got err on {i+1}'s loop")
                print(e)

        img = PIL.Image.open(BytesIO(res[1]))
        metadata = MetadataField(img.info)
        return ImageOutput.build(c.images.save(image=img, metadata=metadata))
