# batdongsan-agentv1

Batdongsan Agent là trợ lý AI cho tìm kiếm, so sánh và phân tích bất động sản. Dự án có 2 cách chạy chính:

- Giao diện web bằng Streamlit: `app_streamlit.py`
- Giao diện dòng lệnh: `scripts/advisor_cli.py`

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu hình

1. Copy file mẫu cấu hình:

```bash
copy CONFIG\global.example.yaml CONFIG\global.yaml
```

2. Cập nhật thông tin kết nối CSDL trong `CONFIG/global.yaml`.
3. Đặt API key cho Gemini bằng một trong hai cách:
	- Thêm key vào file `key.txt` ở thư mục gốc.
	- Hoặc set biến môi trường `GEMINI_API_KEY` / `GOOGLE_API_KEY`.

Mặc định, hệ thống sẽ đọc `CONFIG/global.yaml` và tìm `key.txt` theo cấu hình `EMBEDDING.key_file`.

## Chạy giao diện web

```bash
streamlit run app_streamlit.py
```

Sau khi mở app, bạn có thể nhập câu hỏi tự nhiên như:

- Tìm căn hộ 3 phòng ngủ ở Thủ Đức dưới 5 tỷ
- So sánh quận 7 và Bình Thạnh cho người đi làm trung tâm
- Khu vực nào phù hợp nếu tôi có 8 tỷ và muốn mua nhà phố

## Chạy CLI

### Chế độ tương tác

```bash
python scripts/advisor_cli.py
```

Các lệnh bên trong CLI:

```text
search <query>        - Tìm kiếm bằng câu hỏi tự nhiên
explain <id> <query>  - Giải thích một listing phù hợp với nhu cầu nào
similar <id>          - Tìm listing tương tự
quit                  - Thoát
```

### Chạy một truy vấn cụ thể

```bash
python scripts/advisor_cli.py --query "Tìm căn hộ 2 phòng ngủ ở Quận 7 dưới 4 tỷ"
```

### Giải thích hoặc tìm listing tương tự

```bash
python scripts/advisor_cli.py --explain batdongsan 12345 --query "Tôi cần căn hộ gần trung tâm"
python scripts/advisor_cli.py --similar batdongsan 12345
```

### Tùy chọn hữu ích

```bash
python scripts/advisor_cli.py --top-k 5 --json --query "Nhà phố ở Bình Thạnh"
python scripts/advisor_cli.py --config CONFIG/global.yaml --query "Căn hộ 3 phòng ngủ"
```

## Ghi chú nhanh

- `app_streamlit.py` là entrypoint chính cho UI.
- `scripts/advisor_cli.py` là entrypoint chính cho CLI.
- `batdongsan_ai_chatbot.py` chỉ là lớp wrapper mỏng quanh runtime của agent.
