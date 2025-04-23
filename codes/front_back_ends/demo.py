'''
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
from modelscope.pipelines import pipeline
import cv2
import gradio as gr
import numpy as np
import re
from gradio.components import Component
from util import check_channels, resize_image, save_images
import json
import argparse


BBOX_MAX_NUM = 8
img_save_folder = 'SaveImages'
load_model = True
model_folder = 'dreamtooth_model/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fp32",
        action="store_true",
        default=False,
        help="Whether or not to use fp32 during inference."
    )
    parser.add_argument(
        "--no_translator",
        action="store_true",
        default=False,
        help="Whether or not to use the CH->EN translator, which enable input Chinese prompt and cause ~4GB VRAM."
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default='font/Arial_Unicode.ttf',
        help="path of a font file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"D:\dachuan\anytext\AnyText\models\9\9.ckpt",
        help="load a specified anytext checkpoint"
    )
    args = parser.parse_args()
    return args


args = parse_args()
infer_params = {
    "model": 'iic/cv_anytext_text_generation_editing',
    "model_revision": 'v1.1.3',
    "use_fp16": not args.use_fp32,
    "use_translator": not args.no_translator,
    "font_path": args.font_path,
}
if args.model_path:
    infer_params['model_path'] = args.model_path
if load_model:
    inference = pipeline('my-anytext-task', **infer_params)


def count_lines(prompt):
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    return len(strs)


def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w/ratio))
        rect_h = min(rect_h, int(h*0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles):
            attempts += 1
            continue
        n_pass += 1
        cv2.fillPoly(img, [rect_pts], 255)
        rectangles.append(rect_pts)
        if n_pass == n:
            break
    print("attempts:", attempts)
    if len(rectangles) != n:
        raise gr.Error(f'Failed in auto generate positions after {attempts} attempts, try again!')
    return img


def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False


def draw_rects(width, height, rects):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for rect in rects:
        x1 = int(rect[0] * width)
        y1 = int(rect[1] * height)
        w = int(rect[2] * width)
        h = int(rect[3] * height)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img


def process(mode, prompt, pos_radio, sort_radio, revise_pos, base_model_path, lora_path_ratio, show_debug, draw_img, rect_img, ref_img, ori_img, img_count, ddim_steps, w, h, strength, cfg_scale, seed, eta, a_prompt, n_prompt, *rect_list):
    n_lines = count_lines(prompt)
    # Text Generation
    if mode == 'gen':
        # create pos_imgs
        if pos_radio == 'Manual-draw(手绘)':
            if draw_img is not None:
                pos_imgs = 255 - draw_img['image']
                if 'mask' in draw_img:
                    pos_imgs = pos_imgs.astype(np.float32) + draw_img['mask'][..., 0:3].astype(np.float32)
                    pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
            else:
                pos_imgs = np.zeros((w, h, 1))
        elif pos_radio == 'Manual-rect(拖框)':
            rect_check = rect_list[:BBOX_MAX_NUM]
            rect_xywh = rect_list[BBOX_MAX_NUM:]
            checked_rects = []
            for idx, c in enumerate(rect_check):
                if c:
                    _xywh = rect_xywh[4*idx:4*(idx+1)]
                    checked_rects += [_xywh]
            pos_imgs = draw_rects(w, h, checked_rects)
        elif pos_radio == 'Auto-rand(随机)':
            pos_imgs = generate_rectangles(w, h, n_lines, max_trys=500)
    # Text Editing
    elif mode == 'edit':
        revise_pos = False  # disable pos revise in edit mode
        if ref_img is None or ori_img is None:
            raise gr.Error('No reference image, please upload one for edit!')
        edit_image = ori_img.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=768)
        h, w = edit_image.shape[:2]
        if isinstance(ref_img, dict) and 'mask' in ref_img and ref_img['mask'].mean() > 0:
            pos_imgs = 255 - edit_image
            edit_mask = cv2.resize(ref_img['mask'][..., 0:3], (w, h))
            pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
            pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        else:
            if isinstance(ref_img, dict) and 'image' in ref_img:
                ref_img = ref_img['image']
            pos_imgs = 255 - ref_img  # example input ref_img is used as pos
    cv2.imwrite('pos_imgs.png', 255-pos_imgs[..., ::-1])
    params = {
        "mode": mode,
        "sort_priority": sort_radio,
        "show_debug": show_debug,
        "revise_pos": revise_pos,
        "image_count": img_count,
        "ddim_steps": ddim_steps,
        "image_width": w,
        "image_height": h,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt,
        "base_model_path": base_model_path,
        "lora_path_ratio": lora_path_ratio
    }
    input_data = {
        "prompt": prompt,
        "seed": seed,
        "draw_pos": pos_imgs,
        "ori_image": ori_img,
    }

    results, rtn_code, rtn_warning, debug_info = inference(input_data, **params)
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f'Done, result images are saved in: {img_save_folder}')
        if rtn_warning:
            gr.Warning(rtn_warning)
    else:
        raise gr.Error(rtn_warning)
    return results, gr.Markdown(debug_info, visible=show_debug)


def create_canvas(w=512, h=512, c=3, line=5):
    image = np.full((h, w, c), 200, dtype=np.uint8)
    for i in range(h):
        if i % (w//line) == 0:
            image[i, :, :] = 150
    for j in range(w):
        if j % (w//line) == 0:
            image[:, j, :] = 150
    image[h//2-8:h//2+8, w//2-8:w//2+8, :] = [200, 0, 0]
    return image


def resize_w(w, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (w, img1.shape[0])), cv2.resize(img2, (w, img2.shape[0]))]


def resize_h(h, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (img1.shape[1], h)), cv2.resize(img2, (img2.shape[1], h))]


is_t2i = 'true'

# 读取 description.txt 文件
def load_description_from_json():
    """
    从环境变量 MODELSCOPE_CACHE 中读取配置文件路径，并加载 description 字段。
    """
    # 获取环境变量 MODELSCOPE_CACHE
    modelscope_cache = os.getenv('MODELSCOPE_CACHE')
    
    # 如果环境变量未设置，使用默认路径
    if modelscope_cache is None:
        json_path = model_folder
        json_path = os.path.join(json_path, "iic/cv_anytext_text_generation_editing/configuration.json")
        print(f"环境变量 MODELSCOPE_CACHE 未设置，使用默认路径: {json_path}")
    else:
        # 拼接完整的 JSON 文件路径
        json_path = os.path.join(modelscope_cache, "iic/cv_anytext_text_generation_editing/configuration.json")
        print(f"从环境变量 MODELSCOPE_CACHE 构建路径: {json_path}")
    
    try:
        # 尝试打开并读取 JSON 文件
        with open(json_path, "r", encoding="utf-8") as file:
            config_data = json.load(file)
        
        # 提取 description 字段
        description_tags = config_data.get("description", [])
        if not isinstance(description_tags, list):
            description_tags = [description_tags]  # 确保结果为列表
        
        return description_tags
    
    except FileNotFoundError:
        return ["Description file not found."]
    except json.JSONDecodeError:
        return ["Failed to parse JSON file."]
    except Exception as e:
        return [f"Error reading description file: {str(e)}"]

# 测试加载描述信息
description_tags = load_description_from_json()


block = gr.Blocks(css='style.css', theme=gr.themes.Soft()).queue()

with open('javascript/bboxHint.js', 'r') as file:
    value = file.read()
escaped_value = json.dumps(value)

with block:
    block.load(fn=None,
               _js=f"""() => {{
               const script = document.createElement("script");
               const text =  document.createTextNode({escaped_value});
               script.appendChild(text);
               document.head.appendChild(script);
               }}""")
    gr.HTML('<div style="text-align: center; margin: 20px auto;"> \
        <img id="banner" src="file/example_images/banner.png" alt="anytext"> <br>  \
        </div>')
    with gr.Row(variant='compact'):
        with gr.Column() as left_part:
            pass
        with gr.Column():
            result_gallery = gr.Gallery(label='Result(结果)', show_label=True, preview=True, columns=2, allow_preview=True, height=600)
            result_info = gr.Markdown('', visible=False)
            gr.Markdown(f'<span style="color:silver;font-size:12px"> \
<strong>模型特长：<br></strong></span>')
            tag_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: left;"><br>'
            for tag in description_tags:
                tag_html += f'<span style="background-color: #624AFF; color: white; padding: 5px 10px; border-radius: 5px;">{tag}</span>'
            tag_html += '<br></div>'
            gr.HTML(tag_html)
        with left_part:
            with gr.Accordion('🕹Instructions(说明)', open=False,):
                with gr.Tabs():
                    gr.Markdown('<span style="color:#3B5998;font-size:20px">文字生成</span>')
                    gr.Markdown('<span style="color:#575757;font-size:16px">在Prompt中输入描述提示词（支持中英文），需要生成的每一行文字用双引号包裹，然后依次手绘指定每行文字的位置，确保生成的图像质量。</span>\
                                    <span style="color:red;font-size:16px">文字位置的绘制至关重要</span>, \
                                    <span style="color:#575757;font-size:16px">请尽量避免随意或过小的标注。每个位置的数量应与文字行数一致，并确保尺寸与对应文字的长宽比例相匹配。</span>\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **手绘方式**：可使用 **手绘（Manual-draw）**、**拖框矩形（Manual-rect）** 或 **随机生成（Auto-rand）** 进行标注。\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **排序规则**：多行文字的位置需按一定规则排序后与文字行对应，**Sort Position** 选项可设定排序方式（优先从上到下或从左到右）。\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **调试与优化**：可在参数设置中启用 **Show Debug** 选项，以在生成结果中观察文字位置和字形图；**Revise Position** 选项可用渲染后文字的外接矩形修正位置，但可能会影响创造性。\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                               **参数调节建议**：\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **图片模糊** → 提高步数\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **不需要文字渲染** → 将 `strength` 调至 0\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **生成图片不符合预期** → 适当降低 `strength`，适当提高 `CFG-Scale`\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **文字消失或不清晰** → 适当降低 `CFG-Scale`\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **风格多样性** → `eta = 1`（更具变化）\
                                </span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **风格稳定** → `eta = 0`（更保守）\
                                </span>')

                    gr.Markdown('<span style="color:#3B5998;font-size:20px">文字编辑</span>')
                    gr.Markdown('<span style="color:#575757;font-size:16px">请上传一张待编辑的图片作为**参考图(Ref)**，然后调整笔触大小后，在参考图上涂抹要编辑的位置，在**Prompt**中输入描述提示词和要修改的文字内容，生成图片。</span>')
                    gr.Markdown('<span style="color:gray;font-size:12px">\
                                **注意事项**：\
                                    - 参考图可为任意分辨率，但处理时 **长边不超过 768**，并且宽高会缩放为 **64 的整数倍**。\
                                    - 合理选择笔触大小，确保涂抹区域覆盖完整的修改部分。\
                                    通过这些优化设置，您可以更精准地控制文字生成和编辑效果，创造出更符合需求的视觉作品。 🎨✨\
                                </span>')
            with gr.Accordion('🛠Parameters(参数)', open=False):
                with gr.Row(variant='compact'):
                    img_count = gr.Slider(label="Image Count(图片数)", minimum=1, maximum=12, value=4, step=1)
                    ddim_steps = gr.Slider(label="Steps(步数)", minimum=1, maximum=100, value=20, step=1)
                with gr.Row(variant='compact'):
                    image_width = gr.Slider(label="Image Width(宽度)", minimum=256, maximum=768, value=512, step=64)
                    image_height = gr.Slider(label="Image Height(高度)", minimum=256, maximum=768, value=512, step=64)
                with gr.Row(variant='compact'):
                    strength = gr.Slider(label="Strength(控制力度)", minimum=0.0, maximum=2.0, value=0.8, step=0.01)
                    cfg_scale = gr.Slider(label="CFG-Scale(CFG强度)", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                with gr.Row(variant='compact'):
                    seed = gr.Slider(label="Seed(种子数)", minimum=-1, maximum=99999999, step=1, randomize=False, value=-1)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                with gr.Row(variant='compact'):
                    show_debug = gr.Checkbox(label='Show Debug(调试信息)', value=False)
                    gr.Markdown('<span style="color:silver;font-size:12px">whether show glyph image and debug information in the result(是否在结果中显示glyph图以及调试信息)</span>')
                a_prompt = gr.Textbox(label="Added Prompt(附加提示词)", value='best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
                n_prompt = gr.Textbox(label="Negative Prompt(负向提示词)", value='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')
            base_model_path = gr.Textbox(label='Base Model Path(基模地址)')
            lora_path_ratio = gr.Textbox(label='LoRA Path and Ratio(lora地址和比例)')
            prompt = gr.Textbox(label="Prompt(提示词)")
            with gr.Tabs() as tab_modes:
                with gr.Tab("🖼Text Generation(文字生成)", elem_id='MD-tab-t2i') as mode_gen:
                    pos_radio = gr.Radio(["Manual-draw(手绘)", "Manual-rect(拖框)", "Auto-rand(随机)"], value='Manual-draw(手绘)', label="Pos-Method(位置方式)", info="choose a method to specify text positions(选择方法用于指定文字位置).")
                    with gr.Row():
                        sort_radio = gr.Radio(["↕", "↔"], value='↕', label="Sort Position(位置排序)", info="position sorting priority(位置排序时的优先级)")
                        revise_pos = gr.Checkbox(label='Revise Position(修正位置)', value=False)
                        # gr.Markdown('<span style="color:silver;font-size:12px">try to revise according to text\'s bounding rectangle(尝试通过渲染后的文字行的外接矩形框修正位置)</span>')
                    with gr.Row(variant='compact'):
                        rect_cb_list: list[Component] = []
                        rect_xywh_list: list[Component] = []
                        for i in range(BBOX_MAX_NUM):
                            e = gr.Checkbox(label=f'{i}', value=False, visible=False, min_width='10')
                            x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-x', visible=False)
                            y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-y',  visible=False)
                            w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-w',  visible=False)
                            h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-h',  visible=False)
                            x.change(fn=None, inputs=x, outputs=x, _js=f'v => onBoxChange({is_t2i}, {i}, "x", v)', show_progress=False, queue=False)
                            y.change(fn=None, inputs=y, outputs=y, _js=f'v => onBoxChange({is_t2i}, {i}, "y", v)', show_progress=False, queue=False)
                            w.change(fn=None, inputs=w, outputs=w, _js=f'v => onBoxChange({is_t2i}, {i}, "w", v)', show_progress=False, queue=False)
                            h.change(fn=None, inputs=h, outputs=h, _js=f'v => onBoxChange({is_t2i}, {i}, "h", v)', show_progress=False, queue=False)

                            e.change(fn=None, inputs=e, outputs=e, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)', queue=False)
                            rect_cb_list.extend([e])
                            rect_xywh_list.extend([x, y, w, h])

                    rect_img = gr.Image(value=create_canvas(), label="Rext Position(方框位置)", elem_id="MD-bbox-rect-t2i", show_label=False, visible=False)
                    draw_img = gr.Image(value=create_canvas(), label="Draw Position(绘制位置)", visible=True, tool='sketch', show_label=False, brush_radius=100)

                    def re_draw():
                        return [gr.Image(value=create_canvas(), tool='sketch'), gr.Slider(value=512), gr.Slider(value=512)]
                    draw_img.clear(re_draw, None, [draw_img, image_width, image_height])
                    image_width.release(resize_w, [image_width, rect_img, draw_img], [rect_img, draw_img])
                    image_height.release(resize_h, [image_height, rect_img, draw_img], [rect_img, draw_img])

                    def change_options(selected_option):
                        return [gr.Checkbox(visible=selected_option == 'Manual-rect(拖框)')] * BBOX_MAX_NUM + \
                                [gr.Image(visible=selected_option == 'Manual-rect(拖框)'),
                                 gr.Image(visible=selected_option == 'Manual-draw(手绘)'),
                                 gr.Radio(visible=selected_option != 'Auto-rand(随机)'),
                                 gr.Checkbox(value=selected_option == 'Auto-rand(随机)')]
                    pos_radio.change(change_options, pos_radio, rect_cb_list + [rect_img, draw_img, sort_radio, revise_pos], show_progress=False, queue=False)
                    with gr.Row():
                        gr.Markdown("")
                        run_gen = gr.Button(value="Run(运行)!", scale=0.3, elem_classes='run')
                        gr.Markdown("")
                    def exp_gen_click():
                        return [gr.Slider(value=512), gr.Slider(value=512)]  # all examples are 512x512, refresh draw_img
                    with gr.Tab("示例"):
                        exp_gen_ch = gr.Examples(
                            [
                                ['太阳神鸟金饰，上写着"天路"', "example_images/wenchuang/pos1.png", "Manual-draw(手绘)", "↕", False, 0.7, 11573584],
                                ['长信宫灯，写着"平安"', "example_images/wenchuang/pos2.png", "Manual-draw(手绘)", "↕", False, 0.9, 69283149],
                                ['水晶白菜，“白菜”', "example_images/wenchuang/pos3.png", "Manual-draw(手绘)", "↕", False, 0.9, 69283149],
                                ['卡通青铜树，上方写着"神树"', "example_images/wenchuang/pos4.png", "Manual-draw(手绘)", "↕", False, 0.9, 69283149],
                                ['卡通青铜面具，头顶刻着"王"', "example_images/wenchuang/pos4.png", "Manual-draw(手绘)", "↔", False, 0.9, 69283149],
                            ],
                            [prompt, draw_img, pos_radio, sort_radio, revise_pos, strength, seed],
                            examples_per_page=5,
                            label=''
                        )
                        exp_gen_ch.dataset.click(exp_gen_click, None, [image_width, image_height])

                with gr.Tab("🎨Text Editing(文字编辑)") as mode_edit:
                    with gr.Row(variant='compact'):
                        ref_img = gr.Image(label='Ref(参考图)', source='upload')
                        ori_img = gr.Image(label='Ori(原图)', scale=0.4)

                    def upload_ref(x):
                        return [gr.Image(type="numpy", brush_radius=100, tool='sketch'),
                                gr.Image(value=x)]

                    def clear_ref(x):
                        return gr.Image(source='upload', tool=None)
                    ref_img.upload(upload_ref, ref_img, [ref_img, ori_img])
                    ref_img.clear(clear_ref, ref_img, ref_img)
                    with gr.Row():
                        gr.Markdown("")
                        run_edit = gr.Button(value="Run(运行)!", scale=0.3, elem_classes='run')
                        gr.Markdown("")
                    with gr.Tab("示例"):
                        gr.Examples(
                            [
                                ['精美的书法作品，上面写着“志” “存” “高” ”远“', "example_images/ref10.jpg", "example_images/edit10.png", 0.7, 98053044],
                                ['“上”，“天”', "example_images/wenchuang/ori1.png", "example_images/wenchuang/edit1.png", 1, 80864653],
                                ['一个漫画，上面写着" "', "example_images/ref14.png", "example_images/edit14.png", 0.7, 94081527],
                                ['一个黄色标志牌，上边写着"不要" 和 "大意"', "example_images/ref3.jpg", "example_images/edit3.png", 0.7, 64010349],
                                ['一个青铜鼎，上面写着"  "和"  "', "example_images/ref4.jpg", "example_images/edit4.png", 0.7, 71139289],
                            ],
                            [prompt, ori_img, ref_img, strength, seed],
                            examples_per_page=5,
                            label=''
                        )
    ips = [prompt, pos_radio, sort_radio, revise_pos, base_model_path, lora_path_ratio, show_debug, draw_img, rect_img, ref_img, ori_img, img_count, ddim_steps, image_width, image_height, strength, cfg_scale, seed, eta, a_prompt, n_prompt, *(rect_cb_list+rect_xywh_list)]
    run_gen.click(fn=process, inputs=[gr.State('gen')] + ips, outputs=[result_gallery, result_info])
    run_edit.click(fn=process, inputs=[gr.State('edit')] + ips, outputs=[result_gallery, result_info])


block.launch(
    server_name='0.0.0.0' if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1",
    share=True,
    root_path=f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv('GRADIO_PROXY_PATH') else ""
)
#block.launch(server_name='0.0.0.0')

