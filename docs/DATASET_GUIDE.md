# Hướng dẫn dữ liệu Triage AI

## Cấu trúc CSV chuẩn

File CSV đầu vào phải chứa các cột sau:

| Cột tiếng Anh | Cột tiếng Việt | Mô tả |
|---------------|----------------|-------|
| `age` | `tuoi` | Tuổi |
| `gender` | `gioi_tinh` | Giới tính (Nam/Nữ/Khác) |
| `heart_rate` | `nhip_tim` | Nhịp tim (bpm) |
| `respiratory_rate` | `nhip_tho` | Nhịp thở (lần/phút) |
| `temperature` | `nhiet_do` | Nhiệt độ (°C) |
| `spo2` | `do_bao_hoa_oxy` | Độ bảo hòa oxy (%) |
| `systolic_bp` | `huyet_ap_tam_thu` | Huyết áp tâm thu (mmHg) |
| `diastolic_bp` | `huyet_ap_tam_truong` | Huyết áp tâm trương (mmHg) |
| `triage_level` | `muc_do_uu_tien` | Nhãn đầu ra (1-5) |

## Các file raw hiện có

- `synthetic_medical_triage.csv`: dataset chính, sạch, có target.
- Các file khác cần mapping hoặc xử lý thêm.

## Tiền xử lý

Chạy:

```bash
python src/01_preprocessing.py --input data/raw/synthetic_medical_triage.csv
```

Sau đó chạy các script huấn luyện.
