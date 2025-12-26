import torch
import clip
from PIL import Image
import argparse
import os
import glob
import shutil 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_image(image_path, target_text):
    """
    ターゲットテキストと、その他のネガティブプロンプトとの確率を計算し、
    ターゲットの確率(0.0~1.0)を返します。
    """
    # 比較対象を設定
    # 以下がtext="a res car"の場合の例
    texts = [target_text, "a blue car", "a orange car", "a yellow car", "a white car", "a black car", "a motorcycle", "a bicycle"]    
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(texts).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
        # ターゲット(index 0)の確率を返す
        return probs[0][0]
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Clip images using CLIP model and copy matches.")
    parser.add_argument("--input_dir", type=str, required=True, help="画像の入力ディレクトリ")
    parser.add_argument("--dest_dir", type=str, default="clipped_dir", help="【追加】条件に合った画像のコピー先ディレクトリ")
    parser.add_argument("--text", type=str, required=True, help="探したい対象の説明 (例: 'a red car')")
    parser.add_argument("--threshold", type=float, default=0.75, help="【追加】コピーする判定のしきい値 (0.0 - 1.0, デフォルト: 0.75)")
    parser.add_argument("--output", type=str, default="results.pt", help="スコア結果の保存ファイル名")
    
    args = parser.parse_args()

    # コピー先ディレクトリがなければ作成
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
        print(f"Created directory: {args.dest_dir}")

    image_files = []
    # 大文字小文字の両方に対応
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images. Start processing...")
    print(f"Target: '{args.text}' (Threshold: {args.threshold})")
    
    results = {}
    copied_count = 0
     
    for image_path in image_files:
        prob = clip_image(image_path, args.text)
        results[image_path] = prob
        
        # --- ここが追加機能 ---
        # スコアがしきい値以上ならコピーする
        if prob >= args.threshold:
            filename = os.path.basename(image_path)
            dest_path = os.path.join(args.dest_dir, filename)
            
            # メタデータを保持したままコピー
            shutil.copy2(image_path, dest_path)
            copied_count += 1
            print(f"[COPY] {filename} (Score: {prob:.4f})")


    torch.save(results, args.output)
    print("-" * 30)
    print(f"Processing complete.")
    print(f"Copied {copied_count} images to '{args.dest_dir}'.")
    print(f"Scores saved to {args.output}")

if __name__ == "__main__":
    main()