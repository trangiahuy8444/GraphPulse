# Thư Mục Image

## Giới thiệu

Thư mục `image/` chứa các hình ảnh, biểu đồ, và visualizations được sử dụng trong documentation và presentation của dự án GraphPulse. Các hình ảnh này minh họa kiến trúc hệ thống, kết quả thực nghiệm, và các khái niệm quan trọng của dự án.

Thư mục này phục vụ mục đích:
- Documentation: Hình ảnh cho README và tài liệu
- Presentation: Slides và posters cho conferences
- Visualization: Minh họa concepts và results

## Các file chính

### `Graphpulse.png`

**Mô tả**: Logo chính thức của dự án GraphPulse.

**Sử dụng**:
- Hiển thị trong README.md ở đầu repository
- GitHub repository profile
- Presentations và papers
- Documentation headers

**Kích thước**: 200x200 pixels (theo README.md)

### `System-overview (4).png`

**Mô tả**: Sơ đồ tổng quan hệ thống GraphPulse, mô tả kiến trúc tổng thể của framework.

**Nội dung minh họa**:
- Kiến trúc tổng thể của GraphPulse framework
- Luồng xử lý dữ liệu từ input (raw networks) đến output (predictions)
- Các components chính:
  - Data preprocessing
  - Topological Data Analysis (TDA)
  - RNN models
  - GNN models
- Mối quan hệ và tương tác giữa các modules

**Sử dụng**:
- Hiển thị trong README.md để giải thích system architecture
- Tham khảo trong papers và presentations
- Documentation cho developers hiểu về workflow
- Visual guide cho new users

**Format**: PNG với high resolution để đảm bảo chất lượng khi zoom

### `TDAExample_2.png`

**Mô tả**: Ví dụ minh họa về Topological Data Analysis (TDA) được áp dụng trong GraphPulse.

**Nội dung minh họa**:
- Mapper graph visualization: Cách TDA Mapper method tạo mapper graph từ graph nodes
- Clustering results: Kết quả clustering nodes dựa trên topological features
- Topological features extraction: Quá trình trích xuất features từ topology
- So sánh giữa raw graph và TDA-processed graph

**Mục đích**:
- Giải thích concept TDA cho người đọc chưa quen
- Minh họa output của TDA processing
- So sánh với raw graph representations
- Visual aid cho understanding TDA workflow

**Sử dụng**:
- Documentation về TDA processing
- Papers giải thích methodology
- Tutorials và workshops

## Cách sử dụng

### Trong Markdown Documentation

```markdown
![GraphPulse Logo](image/Graphpulse.png)

![System Overview](image/System-overview%20(4).png)

![TDA Example](image/TDAExample_2.png)
```

**Lưu ý**: Đối với file có spaces trong tên, sử dụng URL encoding (`%20` cho space) hoặc đổi tên file.

### Trong README Files

```markdown
<p align="center">
  <img width="200" height="200" src="image/Graphpulse.png">
</p>

![GraphPulse System Architecture](image/System-overview%20(4).png)
*Graphpulse system overview*
```

### Trong Code Comments

Các hình ảnh có thể được tham chiếu trong code comments để giải thích concepts:

```python
# Áp dụng TDA Mapper method (xem image/TDAExample_2.png)
# để extract topological features từ graph nodes
mapper = km.KeplerMapper()
```

### Trong Presentations

- Export ở resolution cao (300 DPI) cho printing
- Sử dụng format PNG cho diagrams
- Có thể convert sang PDF nếu cần vector format

## Tạo Hình Ảnh Mới

### Naming Convention

1. **Sử dụng tên mô tả**: Tên file nên mô tả nội dung
   - ✅ Good: `training_curves.png`, `model_comparison.png`, `auc_results.png`
   - ❌ Avoid: `image1.png`, `fig.png`, `untitled.png`

2. **Không dùng spaces**: Sử dụng underscores hoặc hyphens
   - ✅ Good: `system_overview.png`, `tda-example.png`
   - ❌ Avoid: `system overview.png`, `TDA Example.png`

3. **Consistent casing**: Sử dụng lowercase hoặc CamelCase nhất quán
   - ✅ Good: `graphpulse.png`, `SystemOverview.png`
   - ❌ Avoid: `GraphPulse.png`, `systemOverview.png` (mixed case)

### Format Recommendations

- **PNG**: Cho diagrams, screenshots, và images với text
  - Lossless compression
  - Hỗ trợ transparency
  - Tốt cho sharp lines và text

- **JPEG**: Cho photos và images với nhiều colors
  - Lossy compression nhưng nhỏ hơn
  - Không hỗ trợ transparency
  - Tốt cho natural images

- **SVG**: Cho vector graphics (nếu cần scale)
  - Vector format, scalable
  - Nhỏ file size
  - Cần trình duyệt hỗ trợ

### Resolution và Quality

- **Documentation**: 300 DPI cho printing quality
- **Web display**: 72-150 DPI đủ cho màn hình
- **Presentations**: 300 DPI cho projectors
- **GitHub README**: Kích thước phù hợp để load nhanh (< 1MB)

### Organization Tips

1. **Subdirectories**: Nếu có nhiều images, tạo subdirectories:
   ```
   image/
   ├── logos/
   ├── diagrams/
   ├── results/
   └── examples/
   ```

2. **Versioning**: Nếu cần nhiều versions:
   - `system-overview-v1.png`, `system-overview-v2.png`
   - Hoặc `system-overview-final.png`

3. **Thumbnails**: Tạo thumbnails cho large images:
   - `system-overview-thumb.png` (nhỏ hơn, load nhanh)

## Lưu ý

### File Size

- **GitHub limits**: Large files (> 100MB) cần Git LFS
- **Load time**: Large images làm chậm README loading
- **Compression**: Nén images trước khi commit
- **Optimization**: Sử dụng tools như `pngquant`, `jpegoptim`

### Accessibility

- **Alt text**: Luôn có alt text trong Markdown
- **Descriptive filenames**: Tên file nên mô tả nội dung
- **Color contrast**: Đảm bảo text readable trên background

### Copyright và Licensing

- **Own images**: Đảm bảo có quyền sử dụng images
- **Attribution**: Nếu dùng images từ nguồn khác, cần attribution
- **License**: Tuân thủ license của images (nếu có)

### Version Control

- **Binary files**: Images là binary files, không diff được
- **Git LFS**: Cân nhắc sử dụng Git LFS cho large files
- **.gitignore**: Có thể ignore generated images nếu cần

### Backup

- **Original sources**: Giữ original sources (PSD, AI files) nếu có
- **Multiple formats**: Có thể giữ cả PNG và SVG versions
- **Documentation**: Document nguồn gốc và tools tạo images

## Best Practices

1. **Consistent style**: Sử dụng cùng style, colors, fonts cho tất cả diagrams
2. **Clear labels**: Đảm bảo text và labels rõ ràng, dễ đọc
3. **High contrast**: Sử dụng colors có contrast cao
4. **Simplified**: Không làm quá phức tạp, focus vào message chính
5. **Updated**: Cập nhật images khi code/architecture thay đổi
