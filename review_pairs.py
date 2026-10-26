"""Create a contact sheet for human review of dHash candidate pairs."""
import argparse
import os
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=Path, default=Path('artifacts/data/near_candidates.json'))
parser.add_argument('--output', type=Path, default=Path('artifacts/data/near_contact.jpg'))
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--limit', type=int, default=9999)
args = parser.parse_args()
root = Path(os.environ.get('FLOWER_DATA_ROOT', 'flower_data'))
pairs = json.loads(args.pairs.read_text())[args.start:args.start + args.limit]
canvas = Image.new('RGB', (700, len(pairs) * 210), 'white')
draw = ImageDraw.Draw(canvas)
for index, pair in enumerate(pairs):
    y = index * 210
    for side, key in enumerate(('left', 'right')):
        path = root / pair[key]
        with Image.open(path) as im:
            thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (330, 175))
        canvas.paste(thumb, (side * 350, y))
        draw.text((side * 350 + 2, y + 178), pair[key], fill='black')
    metric = f"d={pair['distance']}" if 'distance' in pair else f"sim={pair['similarity']:.4f}"
    draw.text((290, y + 195), metric, fill='black')
canvas.save(args.output, quality=85)
