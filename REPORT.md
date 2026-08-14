# BÁO CÁO THỰC HÀNH LAB 16: CLOUD AI ENVIRONMENT SETUP

- **Họ và tên:** Trịnh Quốc Trọng
- **Mã sinh viên:** 2A202601779
- **Môn học / Lớp:** Track 2 - Day 16
- **Cloud Provider:** Amazon Web Services (AWS)
- **Region:** us-east-1

---

## 1. Kết quả Huấn luyện và Đánh giá LightGBM

| Metric | Kết quả |
|---|---|
| **Thời gian nạp dữ liệu (Load Data)** | ~2.95 giây |
| **Thời gian huấn luyện (Training Time)** | ~1.79 giây |
| **Best iteration** | 1 |
| **AUC-ROC** | 0.951654 |
| **Độ chính xác (Accuracy)** | 99.89% (0.998947) |
| **F1-Score** | 0.727273 |
| **Precision** | 0.655738 |
| **Recall** | 0.816327 |
| **Độ trễ suy luận (Inference Latency - 1 dòng)** | 1.2385 ms |
| **Thông lượng suy luận (Inference Throughput - 1000 dòng)** | 648,490.19 rows/giây |

---

## 2. Báo cáo nhận xét (5 - 10 dòng)

1. **Hiệu năng huấn luyện:** Mô hình LightGBM trên máy ảo CPU `t3.medium` (2 vCPU, 4GB RAM) xử lý 284,807 bản ghi dữ liệu cực kỳ nhanh, hoàn tất huấn luyện trong chưa đầy 2 giây (~1.79s).
2. **Chất lượng dự báo:** Mặc dù bộ dữ liệu giao dịch gian lận (Credit Card Fraud) bị mất cân bằng nghiêm trọng (tỷ lệ gian lận rất nhỏ), mô hình vẫn đạt điểm AUC-ROC ấn tượng **0.9516** và độ nhạy (Recall) **81.63%**, đảm bảo phát hiện phần lớn các giao dịch khả nghi.
3. **Tốc độ suy luận (Inference Speed):** Thời gian phản hồi cho từng giao dịch đơn lẻ chỉ **~1.24 ms**, và khả năng xử lý lô lớn đạt hơn **648,000 giao dịch/giây**. Điều này chứng minh giải pháp hoàn toàn đáp ứng tốt các bài toán kiểm soát giao dịch thời gian thực (real-time) với chi phí hạ tầng thấp (~$0.10/giờ) mà chưa cần đến GPU đắt đỏ.

---

## 3. Danh sách hình ảnh minh chứng

1. `screenshots/01_benchmark_result.png`: Ảnh chụp terminal chạy `benchmark.py` hiển thị bảng kết quả.
2. `screenshots/02_resource_monitoring.png`: Ảnh chụp mức sử dụng CPU (`top`), RAM (`free -h`), và Network (`ip -s link`).
3. `screenshots/03_aws_billing.png`: Ảnh chụp AWS Billing / Cost Dashboard ghi nhận chi phí EC2 & NAT Gateway.
