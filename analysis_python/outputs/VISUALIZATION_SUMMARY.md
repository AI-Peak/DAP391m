
# MULTI-CHANNEL ATTRIBUTION ANALYSIS - VISUALIZATION SUMMARY

## Hình 1: Attribution Share by Method (So sánh 3 phương pháp)

**Loại biểu đồ:** Grouped Bar Chart (Biểu đồ cột nhóm)

### Mục đích
So sánh cách ba mô hình Attribution phân bổ credit cho từng marketing channel.

### Các mô hình được sử dụng

- **First-Touch Attribution:** Toàn bộ credit được gán cho điểm chạm đầu tiên khi khách hàng lần đầu tiếp xúc với thương hiệu.
- **Last-Touch Attribution:** Toàn bộ credit được gán cho điểm chạm cuối cùng trước khi khách hàng chuyển đổi.
- **Linear Attribution:** Credit được chia đều cho tất cả các điểm chạm trong hành trình khách hàng.

### Ý nghĩa
Biểu đồ cho thấy sự khác biệt trong cách mỗi phương pháp đánh giá mức độ đóng góp của từng channel đối với conversion.

---

## Hình 2: Channel Conversion Rate (Tỷ lệ chuyển đổi của từng Channel)

**Loại biểu đồ:** Horizontal Bar Chart (Biểu đồ cột ngang)

### Mục đích
Đánh giá hiệu quả chuyển đổi của từng marketing channel.

### Hiển thị

- Tỷ lệ chuyển đổi (%) của từng channel.
- Các channel được sắp xếp từ thấp đến cao.

### Ý nghĩa
Giúp xác định channel nào có khả năng chuyển đổi khách hàng tốt nhất và channel nào hoạt động kém hiệu quả hơn.

---

## Hình 3: Logit-Adjusted Attribution Share (Kết quả hồi quy Logistic)

**Loại biểu đồ:** Bar Chart (Biểu đồ cột)

### Mục đích
Đánh giá mức độ ảnh hưởng của từng channel bằng mô hình Machine Learning.

### Phương pháp

Sử dụng mô hình **Logistic Regression** để:

- Phân tích dữ liệu lịch sử.
- Ước lượng mức độ ảnh hưởng của từng channel đến xác suất chuyển đổi.
- Điều chỉnh Attribution Share dựa trên sức ảnh hưởng thực tế.

### Ý nghĩa
Các channel có Attribution Share cao hơn được xem là có ảnh hưởng mạnh hơn đến conversion.

---

## Hình 4: Model Comparison (Biểu đồ quan trọng nhất của RQ2)

**Loại biểu đồ:** Heatmap (Bản đồ nhiệt)

### Mục đích
So sánh kết quả Attribution giữa năm mô hình khác nhau.

### Các mô hình được so sánh

1. First-Touch
2. Last-Touch
3. Linear
4. Logistic Regression (Logit)
5. Markov Chain

### Quan sát chính

#### Display Ads

- Được đánh giá rất cao bởi các mô hình hiện đại.
- Logit Attribution: 21.99%
- Markov Attribution: 76.06%

Điều này cho thấy Display Ads có vai trò quan trọng trong quá trình dẫn dắt khách hàng đến conversion.

#### Social Media

- Attribution Share tương đối ổn định trên tất cả các mô hình.
- Dao động khoảng 16%–17%.

Điều này cho thấy Social Media có mức đóng góp nhất quán và ít gây tranh cãi giữa các phương pháp đánh giá.

#### Linear Attribution

- Các giá trị phân bổ nằm ở mức trung bình.
- Không quá thiên vị điểm chạm đầu tiên hay cuối cùng.
- Được xem là phương pháp công bằng nhất.

### Kết luận cho RQ2

- **Linear Attribution** được khuyến nghị khi cần sự cân bằng và dễ giải thích.
- **Logit** hoặc **Markov** phù hợp hơn khi doanh nghiệp muốn tối đa hóa độ chính xác trong phân tích hành vi khách hàng.

---

## Hình 5: Markov Removal Effect (Tác động khi loại bỏ Channel)

**Loại biểu đồ:** Horizontal Bar Chart (Biểu đồ cột ngang)

### Mục đích

Đo lường mức thay đổi của tỷ lệ chuyển đổi nếu một channel bị loại khỏi customer journey.

### Câu hỏi được trả lời

"Nếu channel này không tồn tại thì conversion sẽ thay đổi như thế nào?"

### Kết quả

#### Display Ads

- Removal Effect: +0.0535%

Nếu loại bỏ Display Ads, tỷ lệ chuyển đổi sẽ giảm đáng kể.

#### Referral

- Removal Effect: +0.0168%

Referral cũng đóng vai trò quan trọng trong việc hỗ trợ chuyển đổi.

#### Social Media, Direct Traffic, Email và Search Ads

- Giá trị âm hoặc gần bằng 0.

Điều này cho thấy việc loại bỏ các channel này không tạo ra sự thay đổi đáng kể đối với conversion rate.

### Kết luận

Display Ads và Referral là hai channel có giá trị hỗ trợ lớn nhất trong customer journey theo góc nhìn Markov Chain.

---

# Hình 6: Simulation Scenario Results (RQ3)

Hình 6 là một Dashboard mô phỏng nhằm so sánh ba chiến lược phân bổ ngân sách marketing khác nhau.

## Ba kịch bản được đánh giá

| Kịch bản | Tên | Mô tả |
|----------|------|--------|
| S0 | Equal Split | Chia đều ngân sách cho 6 channel, mỗi channel nhận 16.67% |
| S1 | Conversion-Rate Weighted | Phân bổ ngân sách theo tỷ lệ chuyển đổi của từng channel |
| S2 | Linear Attribution Weighted | Phân bổ ngân sách theo kết quả của mô hình Linear Attribution |

---

## Ô 1: Delta Revenue vs Baseline (S0)

### Kết quả

| Scenario | Delta Revenue |
|-----------|---------------|
| S0 | $0.00 |
| S1 | +$99.71 |
| S2 | +$82.72 |

### Giải thích

- S1 tạo ra mức tăng doanh thu cao nhất.
- S2 cũng tạo ra cải thiện nhưng thấp hơn S1.

Nguyên nhân là S1 tập trung ngân sách nhiều hơn vào các channel có conversion rate cao như Email và Referral.

---

## Ô 2: Delta Revenue Percentage vs Baseline (S0)

### Kết quả

| Scenario | Delta Revenue (%) |
|-----------|------------------|
| S0 | 0.0000% |
| S1 | +0.0202% |
| S2 | +0.0167% |

### Giải thích

Cả S1 và S2 đều cải thiện doanh thu nhưng mức tăng tương đối nhỏ.

Nguyên nhân là:

- Các channel trong bộ dữ liệu có hiệu suất khá tương đồng.
- Việc thay đổi phân bổ ngân sách chỉ tạo ra lợi ích biên (marginal gain).

Ví dụ:

- Doanh thu cơ sở khoảng $494,000.
- Tăng 0.0202% tương ứng khoảng $99.71.

---

## Ô 3: Delta Conversions vs Baseline (S0)

### Kết quả

| Scenario | Delta Conversions |
|-----------|------------------|
| S0 | 0 |
| S1 | +0.997 |
| S2 | +0.827 |

### Giải thích

- S1 tạo thêm khoảng 1 conversion.
- S2 tạo thêm khoảng 0.8 conversion.

Mặc dù mức tăng không lớn nhưng vẫn mang lại lợi ích tài chính tích cực khi áp dụng trên quy mô lớn.

---

## Ô 4: Total Conversions Across Scenarios

### Kết quả

| Scenario | Total Conversions |
|-----------|------------------|
| S0 | 4,943 |
| S1 | 4,944 |
| S2 | 4,944 |

### Giải thích

Tổng số conversion giữa ba kịch bản gần như tương đương nhau.

Điều này cho thấy:

- Không có chiến lược nào tạo ra sự đột phá lớn.
- Các thay đổi chủ yếu là tối ưu hóa biên.
- Hiệu quả của việc tái phân bổ ngân sách tương đối hạn chế trong bộ dữ liệu hiện tại.

---

# Ý nghĩa Quản trị của Hình 6

Hình 6 là phần quan trọng nhất của toàn bộ dự án vì nó chuyển các kết quả phân tích học thuật thành giá trị kinh doanh cụ thể.

Thay vì chỉ xác định channel nào quan trọng, hình này trả lời trực tiếp câu hỏi:

> "Nếu thay đổi cách phân bổ ngân sách theo mô hình đề xuất, doanh nghiệp sẽ kiếm thêm bao nhiêu tiền?"

## 1. Hỗ trợ xây dựng chiến lược phân bổ ngân sách

Hình 6 cung cấp sẵn các kịch bản thay thế để so sánh với phương pháp hiện tại.

### S1: Conversion-Rate Weighted

Nguyên tắc:

> Đầu tư nhiều hơn vào những channel có khả năng tạo conversion cao nhất.

Ví dụ:

- Email
- Referral

### S2: Linear Attribution Weighted

Nguyên tắc:

> Phân bổ ngân sách dựa trên mức đóng góp của các điểm chạm trong customer journey.

Ví dụ:

- Display Ads
- Direct Traffic

---

## 2. Định lượng hóa lợi ích tài chính (ROI)

Hình 6 giúp chuyển các kết quả Data Science thành giá trị tài chính cụ thể.

Ví dụ:

- S1 tạo thêm khoảng $99.71 doanh thu.
- Tương đương mức tăng khoảng 0.02%.

Nhờ đó:

- Marketing Team có cơ sở ra quyết định.
- Ban Giám Đốc có căn cứ để phê duyệt chiến lược ngân sách mới.

---

## 3. Đánh giá mức độ rủi ro

Mặc dù S1 và S2 cho kết quả tốt hơn S0, mức cải thiện vẫn rất nhỏ.

### Quan sát

- Doanh thu chỉ tăng khoảng 0.02%.
- Tổng conversion chỉ tăng khoảng 1 khách hàng.

### Ý nghĩa

Điều này giúp doanh nghiệp:

- Tránh kỳ vọng quá mức.
- Hiểu rằng tối ưu hóa marketing attribution thường là bài toán tối ưu biên.
- Thực hiện thử nghiệm trên quy mô nhỏ trước khi áp dụng rộng rãi.

---

# Kết luận

Hình 6 là công cụ hỗ trợ ra quyết định quan trọng nhất trong nghiên cứu.

Nó giúp thuyết phục nhà quản lý bằng cách trả lời trực tiếp:

> "Nếu áp dụng các kết quả từ Data Science vào việc phân bổ ngân sách marketing, doanh nghiệp sẽ thu được thêm bao nhiêu doanh thu so với cách phân bổ truyền thống?"

Đồng thời, hình cũng cung cấp một góc nhìn thực tế về mức độ cải thiện có thể đạt được, giúp doanh nghiệp cân bằng giữa kỳ vọng lợi nhuận và rủi ro triển khai.

