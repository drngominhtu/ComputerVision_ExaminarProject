import easyocr

# Danh sách từ cấm (có thể thay bằng đọc file nếu muốn)
with open('banned_words.txt', 'r', encoding='utf-8') as f:
    banned_words = [line.strip().lower() for line in f]

def check_banned_words(text, banned_list):
    found = []
    words = text.lower().split()
    for word in words:
        if word in banned_list:
            found.append(word)
    return set(found)

def main():
    image_path = 'test_images/sample1.jpg'  # Thay ảnh của bạn ở đây
    print("[INFO] Đang quét ảnh:", image_path)

    reader = easyocr.Reader(['vi', 'en'])  # Hỗ trợ tiếng Việt và Anh
    results = reader.readtext(image_path)

    full_text = ' '.join([text for _, text, _ in results])
    print("\n[INFO] Văn bản trích xuất:")
    print(full_text)

    flagged = check_banned_words(full_text, banned_words)

    if flagged:
        print("\n🚨 Phát hiện từ bị cấm:", ', '.join(flagged))
    else:
        print("\n✅ Không phát hiện từ bị cấm.")

if __name__ == "__main__":
    main()
