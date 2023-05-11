python style_transfer.py --content_path $1 --style_path $2 --output_path style_1.jpg
python run.py style_1.jpg $2 --output_path remix.jpg