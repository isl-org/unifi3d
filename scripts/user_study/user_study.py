import os
import sys
import socket
import numpy as np
from pathlib import Path

try:
    from nicegui import app, ui
except ImportError:
    print("No nicegui installed, please run `pip install nicegui`. Exiting now")
    exit()
from datetime import datetime
import random

import argparse
from tempfile import TemporaryDirectory
import shutil
import json
import random
import time

# this collects all submitted data
RESULTS = []

MAX_AMOUNT_TO_COMPARE = 2


def patch_html():
    """This needs to be called for each page to enable the model viewer"""
    ui.add_body_html(
        '<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>'
    )
    ui.add_body_html(
        """
<script>
function toggleMaterial(el_id) {
    mv = document.getElementById(el_id).children[0];
    mat = mv.model.materials[0];

    const textures = ['baseColor', 'metallicRoughness'];

    for (let x of textures ){
        if (! mv.hasAttribute('toggle_'+x)) {
            mv.setAttribute('toggle_'+x, null);
        }
        tmp = mat.pbrMetallicRoughness[x+'Texture'].texture;
        mat.pbrMetallicRoughness[x+'Texture'].setTexture(mv['toggle_'+x]);
        mv['toggle_'+x] = tmp;
    }
}
function dismissPoster(el_id) {
    mv = document.getElementById(el_id).children[0];
    mv.dismissPoster();
}
</script>"""
    )


def model_viewer(
    source: str,
    width=None,
    height=None,
    show_controls=False,
    lazy_load=False,
    size_hint=None,
    poster=None,
):
    """Creates the model viewer
    Args:
        source: Path to the 3d model
        width: width in px
        height: height in px
        show_controls: Show controls to toggle materials
        lazy_load: Show a button for loading the model
        size_hint: size hint string that will be shown next to the load buttong
        poster: Path to the poster
    """
    style = []
    if width:
        style.append(f"width:{width}px")
    if height:
        style.append(f"height:{height}px")
    style = 'style="{}"'.format("; ".join(style))
    controls = ""
    controls = """<div class="controls">"""
    if lazy_load:
        if not size_hint:
            size_hint = ""
        controls += f"""<button class="p-1 bg-slate-300" style="position: absolute; top: 50%; left:50%; transform: translate(-50%, -50%);" onClick="this.style.display='none'; dismissPoster(this.parentElement.parentElement.parentElement.id)">Load model {size_hint}</button>"""
    if show_controls:
        controls += f"""<button class="p-1 bg-slate-300" onClick="toggleMaterial(this.parentElement.parentElement.parentElement.id)">Mat.</button>"""
    controls += """</div>"""

    reveal = ""
    if lazy_load:
        reveal = 'reveal="manual"'

    if poster:
        poster = f'poster="{poster}"'
    else:
        poster = ""

    x = f"""<model-viewer {style} src="{source}" shadow-intensity="1" camera-controls touch-action="pan-y" {reveal} {poster}>{controls}</model-viewer>"""
    return ui.html(x)


def patch():
    """Call this function once for the process to register the model viewer"""
    if not hasattr(ui, "model_viewer"):
        ui.model_viewer = model_viewer
        ui.model_viewer.patch_html = patch_html


def parse_args():
    parser = argparse.ArgumentParser(
        description="User study for unifi3d",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "study_dir",
        type=Path,
        help="Path to the directory created with 'prepare_user_study_data.py'",
    )
    parser.add_argument("--port", type=int, default=8081, help="Port")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    return args


patch()

args = parse_args()
DATA_PATH = args.study_dir / "data"
SAVE_PATH = args.study_dir / "user_results"
CSV_SAVE_PATH = args.study_dir / "user_results" / "csv_data"
CSV_SAVE_PATH.mkdir(parents=True, exist_ok=True)

if (SAVE_PATH / "user_study_latest.json").exists():
    with open(SAVE_PATH / "user_study_latest.json", "r") as f:
        RESULTS = json.load(f)


app.add_static_files("/temp", DATA_PATH)


def init_comparisons_dict():
    method_dirs = list(
        filter(lambda x: not x.name.startswith("."), DATA_PATH.iterdir())
    )
    comparisons = {}

    def make_relative(x):
        return str(x.relative_to(DATA_PATH))

    for md in method_dirs:
        comparisons[md.name] = {
            "paths": sorted(list(map(make_relative, md.glob("*.glb"))))
        }
    return comparisons


def save_results(name=None):
    date = ":".join(
        datetime.now().isoformat().split(":")[:-1]
    )  # isoformat without seconds

    if RESULTS:
        print(f"Saving results: {RESULTS}")

        with open(
            # f"{SAVE_PATH}/user_study_{name}_{name_date}.json", "w"
            # f"{SAVE_PATH}/user_study_{name}.json", "w"
            f"{SAVE_PATH}/user_study_{date}.json",
            "w",
        ) as f:
            json.dump(RESULTS, f, indent=4)
        with open(f"{SAVE_PATH}/user_study_latest.json", "w") as f:
            json.dump(RESULTS, f, indent=4)


COMPARISONS = init_comparisons_dict()
COMPARISONS_KEYS = list(COMPARISONS.keys())

AMOUNT_QUESTIONS = 25


def header(str):
    with ui.header().style("background-color: rgb(0 84 174)"):
        ui.markdown(str)


@ui.page("/thankyou")
def test_page():
    with ui.row().classes("fixed-center"):
        ui.markdown("# Thank You!")


@ui.page("/form")
def form_page():
    ui.model_viewer.patch_html()
    header("# User study")

    with ui.row().classes("w-full"):
        progressbar = ui.linear_progress(show_value=False)

    with ui.row():

        def name_fn(e):
            if len(e.value.strip()) >= 1:
                start_btn.enable()

        user_name = ui.input(label="Enter your name", on_change=name_fn)
        ui.markdown("#### No underscores, no spaces or other weird characters!")

    def end_study():
        save_results(user_name.value.strip())
        ui.navigate.to("/thankyou")

    with ui.column():
        prompt_label = ui.markdown("")
        prompt_container = ui.row()
        # ui.button("End study!", on_click=lambda: end_study())

    class State:
        def __init__(self):
            self.data_idx = 0
            self.progress = 0.0

    current_state = State()

    container = ui.row()

    def submit_and_next():
        nonlocal current_state
        user_name.disable()
        container.clear()
        prompt_container.clear()
        instructions = "## Chose the best 3D model\n(may need time to load)"
        name = user_name.value.strip()
        current_state.data_idx = len([x for x in RESULTS if x["user"] == name])
        current_state.progress = current_state.data_idx / AMOUNT_QUESTIONS
        progressbar.set_value(current_state.progress)

        save_results(user_name.value.strip())

        if current_state.data_idx >= AMOUNT_QUESTIONS:
            ui.navigate.to("/thankyou")
            return

        with container:
            # Choose AMOUNT_TO_COMPARE methods:
            assert len(COMPARISONS_KEYS) >= 2

            keys_to_compare = random.sample(
                COMPARISONS_KEYS, min(MAX_AMOUNT_TO_COMPARE, len(COMPARISONS_KEYS))
            )
            # randomly select one asset from method

            cmps = []
            for key in keys_to_compare:
                cmp = COMPARISONS[key]
                # asset = random.choice(cmp["paths"])
                asset = cmp["paths"][current_state.data_idx % len(cmp["paths"])]
                cmps.append((key, asset))

            random.shuffle(cmps)

            def finalize_submission(event):
                nonlocal current_state
                key, asset = event.sender.key_asset

                current_state.data_idx += 1
                current_state.progress = current_state.data_idx / AMOUNT_QUESTIONS
                progressbar.set_value(current_state.progress)

                RESULTS.append(
                    {
                        "user": user_name.value.strip(),
                        "best_method": key,
                        "compared_methods": keys_to_compare,
                        "best_asset": asset,
                        "compared_assets": [x[1] for x in cmps],
                    }
                )

                submit_and_next()

            prompt_label.set_content(instructions)

            for cmp in cmps:
                key, asset = cmp
                with ui.card().tight() as card:
                    card.classes("bg-slate-100")
                    with ui.card_section():

                        model_viewer(
                            f"/temp/{asset}?{time.time()}",  # append timestamp to avoid the browser using the cache
                            width=300,
                            height=300,
                        )

                        # Create a group of radio buttons for ranking
                        ui.button(
                            "This is the best asset!",
                            on_click=finalize_submission,
                        ).key_asset = (key, asset)

    with container:
        start_btn = ui.button("Start", icon="send", on_click=submit_and_next)
        start_btn.disable()


def create_pairwise_prefs(results):
    lines = []
    for x in results:
        best_m = x["best_method"]
        for m in x["compared_methods"]:
            if m != best_m:
                lines.append(f"{best_m},{m},0")
    return "\n".join(lines)


@ui.page("/result")
def result_page():
    header("# Results")
    data = []
    for x in RESULTS:
        data.append(str(x))

    with open(CSV_SAVE_PATH / "user_study_pairwise_prefs.csv", "w") as f:
        f.write(create_pairwise_prefs(RESULTS))
        ui.label(
            f'pairwise prefs written to {str(CSV_SAVE_PATH/"user_study_pairwise_prefs.csv")}'
        )

    def download_results_fn():
        ui.download(
            create_pairwise_prefs(RESULTS).encode("utf-8"),
            "user_study_pairwise_prefs.csv",
        )

    ui.button("Download pairwise prefs", icon="download", on_click=download_results_fn)
    ui.code("\n".join(data)).classes("w-full")


print(f"http://{socket.getfqdn()}:{args.port}")
ui.run(title="User study", show=False, reload=False, port=args.port)
