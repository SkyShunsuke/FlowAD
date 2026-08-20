from pypdf import PdfReader, PdfWriter, Transformation

# ==============================
# 設定
# ==============================

input_files = [
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/arch/combined_mad.pdf",
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/framework/combined_mad.pdf",
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/latent/combined_mad.pdf",
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/noise_levels/combined_mad.pdf",
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/noise_levels/combined_mad.pdf",
    "/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/noise_levels/combined_mad.pdf",
]

output_file = "combined_2x3.pdf"

# A4 landscape [pt]
# 1 pt = 1/72 inch
PAGE_WIDTH = 841.89
PAGE_HEIGHT = 450.28

ROWS = 2
COLS = 3

# セル間の余白 [pt]
MARGIN = 0


# ==============================
# 出力ページを作成
# ==============================

writer = PdfWriter()

output_page = writer.add_blank_page(
    width=PAGE_WIDTH,
    height=PAGE_HEIGHT,
)

cell_width = PAGE_WIDTH / COLS
cell_height = PAGE_HEIGHT / ROWS


# ==============================
# 6つのPDFを配置
# ==============================

for i, pdf_file in enumerate(input_files):
    reader = PdfReader(pdf_file)

    # 各PDFの1ページ目を使用
    page = reader.pages[0]

    src_width = float(page.mediabox.width)
    src_height = float(page.mediabox.height)

    # セル内にアスペクト比を維持して収める
    available_width = cell_width - 2 * MARGIN
    available_height = cell_height - 2 * MARGIN

    scale = min(
        available_width / src_width,
        available_height / src_height,
    )

    scaled_width = src_width * scale
    scaled_height = src_height * scale

    # 配置位置
    row = i // COLS
    col = i % COLS

    # PDF座標系は左下原点なので、
    # 上段から順に並ぶようにY座標を計算
    cell_x = col * cell_width
    cell_y = PAGE_HEIGHT - (row + 1) * cell_height

    # 各セルの中央に配置
    tx = cell_x + (cell_width - scaled_width) / 2
    ty = cell_y + (cell_height - scaled_height) / 2

    transform = (
        Transformation()
        .scale(scale)
        .translate(tx, ty)
    )

    output_page.merge_transformed_page(
        page,
        transform,
    )


# ==============================
# 保存
# ==============================

with open(output_file, "wb") as f:
    writer.write(f)

print(f"Saved: {output_file}")