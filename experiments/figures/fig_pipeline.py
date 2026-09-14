"""Figura: esquema do pipeline detectar-recortar-refinar. Sem dados, so o diagrama."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BOXES = ["Full-disk H$\\alpha$ image\n$2048 \\times 2048$",
         "YOLO11 detector\n($1024^2$ box or\n$2048^2$ instance)",
         "Padded crop\n$+$ letterbox\n($256 \\times 256$)",
         "EdgeAtt U-Net refiner\n($+$ flip-4 TTA,\ntopology-safe)",
         "Undo letterbox,\npaste mask back\nat full resolution",
         "Mask-level NMS\n(greedy, highest\nconfidence first)",
         "Instance label map\n$+$ RLE submission"]

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 46)
W, H = 20.5, 15.0
pos = [(1.5, 28), (26, 28), (50.5, 28), (75, 28), (75, 4), (50.5, 4), (26, 4)]
for (x, y), t in zip(pos, BOXES):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.5,rounding_size=1.6",
                                 fc="#f4f7fb", ec="#1f4e79", lw=1.0))
    ax.text(x + W / 2, y + H / 2, t, ha="center", va="center", fontsize=7.6)

def arrow(x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                  mutation_scale=11, lw=1.1, color="#1f4e79"))
for i in range(3):                                   # linha de cima, esquerda -> direita
    arrow(pos[i][0] + W, pos[i][1] + H / 2, pos[i + 1][0] - 0.6, pos[i + 1][1] + H / 2)
arrow(pos[3][0] + W / 2, pos[3][1], pos[4][0] + W / 2, pos[4][1] + H + 0.6)   # desce
for i in range(4, 6):                                # linha de baixo, direita -> esquerda
    arrow(pos[i][0], pos[i][1] + H / 2, pos[i + 1][0] + W + 0.6, pos[i + 1][1] + H / 2)
fig.tight_layout(pad=0.2)
out = "Marchezi_etal_2026_SolarPhysics_FilamentSegmentation/figures/pipeline.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("salvo:", out)
