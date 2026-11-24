"""Vietnamese presentation dashboard for the integrated segmentation project."""

from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from demo_data import ACTIONS, COLORS, RFM, SEGMENTS, filter_customers, load_dashboard, load_models, predict_scenario

st.set_page_config(page_title="Retail Insight | Phân khúc khách hàng", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1420px; padding-top: 2.5rem; padding-bottom: 3rem;}
h1 {letter-spacing: -.035em; font-weight: 750;}
h2, h3 {letter-spacing: -.02em;}
[data-testid="stMetric"] {background: white; border: 1px solid #E3EAF3; border-radius: 14px; padding: 18px 20px;}
[data-testid="stMetricValue"] {font-weight: 700;}
[data-testid="stSidebar"] {border-right: 1px solid #E3EAF3;}
.eyebrow {color: #168C7F; font-size: .85rem; font-weight: 700; letter-spacing: .13em; margin-bottom: .4rem;}
.lead {font-size: 1.1rem; color: #52637C; max-width: 850px; line-height: 1.7;}
.takeaway {background: #E9F5F2; border-left: 4px solid #168C7F; padding: 16px 20px; border-radius: 0 10px 10px 0; color: #214E47; line-height: 1.7; margin: 14px 0 22px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_data(stamp):
    return load_dashboard(ROOT)


@st.cache_resource(show_spinner=False)
def cached_models(stamp):
    return load_models(ROOT)


def chart(chart_object):
    st.altair_chart(chart_object.configure_view(stroke=None).configure_axis(
        gridColor="#E8EDF4", labelFontSize=12, titleFontSize=13
    ), width="stretch")


def color_encoding(field="Phân khúc:N"):
    return alt.Color(field, scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=alt.Legend(title=None))


def takeaway(text):
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown('<div class="eyebrow">RETAIL INSIGHT / DATA MINING</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="lead">{subtitle}</p>', unsafe_allow_html=True)


def share_chart(profiles):
    melted = profiles.melt(id_vars=["Phân khúc"], value_vars=["CustomerShare", "MonetaryShare"],
                            var_name="Chỉ tiêu", value_name="Tỷ trọng")
    melted["Chỉ tiêu"] = melted["Chỉ tiêu"].map({"CustomerShare": "Khách hàng", "MonetaryShare": "Chi tiêu ròng"})
    return alt.Chart(melted).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Tỷ trọng:Q", axis=alt.Axis(format=".0%"), title="Tỷ trọng trong toàn bộ tập khách"),
        y=alt.Y("Phân khúc:N", sort=list(COLORS), title=None, axis=alt.Axis(labelLimit=210)),
        yOffset="Chỉ tiêu:N", color=alt.Color("Chỉ tiêu:N", scale=alt.Scale(
            domain=["Khách hàng", "Chi tiêu ròng"], range=["#A9BBD7", "#168C7F"]),
            legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Phân khúc:N", "Chỉ tiêu:N", alt.Tooltip("Tỷ trọng:Q", format=".2%")],
    ).properties(height=310)


def overview(data):
    customers, profiles, audit = data["customers"], data["profiles"], data["audit"]
    header("Phân khúc khách hàng bằng RFM và phân cụm", "Từ giao dịch bán hàng đến bốn nhóm khách có đặc điểm và hướng chăm sóc khác nhau.")
    a, b, c, d = st.columns(4)
    a.metric("Khách hàng phân tích", f"{len(customers):,}")
    b.metric("Chi tiêu ròng trong kỳ", f"£{customers.Monetary.sum()/1e6:.2f} triệu")
    c.metric("Phân khúc K-Means", "4 nhóm")
    vip = profiles.loc[profiles.Segment.eq("VIP")].iloc[0]
    d.metric("Chi tiêu từ nhóm VIP", f"{vip.MonetaryShare:.1%}")
    st.caption("Các chỉ số trên mô tả toàn bộ tập khách hàng sau xử lý. Đơn vị tiền: GBP.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Bài toán cần giải quyết")
        st.write("Khách hàng khác nhau về thời điểm mua, tần suất và mức chi tiêu. Nếu chăm sóc tất cả theo cùng một cách, doanh nghiệp khó xác định nhóm cần giữ chân và nhóm cần tái kích hoạt.")
        st.markdown("**Câu hỏi nghiên cứu**\n\n- Có thể nhận diện những nhóm hành vi nào từ RFM?\n- Phương pháp nào tạo nhóm ổn định và dễ giải thích?\n- Mỗi nhóm gợi ý hành động chăm sóc nào?")
        st.info("Học không giám sát: không có nhãn khách hàng chuẩn để tính accuracy. Các chiến lược là giả thuyết cần thử nghiệm.")
    with right:
        st.subheader("Số khách và giá trị đóng góp")
        chart(share_chart(profiles))
    takeaway(f"<b>{vip.CustomerShare:.1%} khách VIP</b> đóng góp <b>{vip.MonetaryShare:.1%} chi tiêu ròng</b>. Đây là nhóm ưu tiên cho chiến lược giữ chân.")
    st.subheader("Hành trình phân tích")
    cols = st.columns(4)
    for col, title, text in zip(cols, ["Dữ liệu", "EDA", "Modeling", "Diễn giải"],
                               ["Làm sạch và tạo RFM", "Khám phá phân phối và ngoại lệ", "So sánh K-Means, GMM, Ward", "Chân dung và đề xuất chăm sóc"]):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
    with st.expander("Nguồn dữ liệu, phạm vi và định nghĩa"):
        st.write(f"Kỳ quan sát: {audit['first_transaction'][:10]} đến {audit['last_transaction'][:10]}.")
        st.markdown("Nguồn: [Chen, D. (2015), UCI Online Retail](https://doi.org/10.24432/C5BW33), CC BY 4.0.")
        st.markdown("**R — Recency:** số ngày từ lần mua gần nhất đến ngày tham chiếu.\n\n**F — Frequency:** số hóa đơn mua khác nhau trong kỳ.\n\n**M — Monetary:** giá trị mua trừ hoàn/hủy ghi nhận trong kỳ.")
        st.caption("Monetary chưa phải lợi nhuận; chưa đối soát từng dòng hoàn với đơn gốc. Tên nhóm mô tả tương đối trong tập dữ liệu.")
    report = ROOT / "docs/REPORT.md"
    if report.exists():
        st.download_button("Tải báo cáo tiếng Việt (.md)", report.read_bytes(), "Bao_cao_phan_khuc.md", "text/markdown")


def eda(data):
    customers = data["customers"]
    header("EDA · Hiểu dữ liệu trước khi phân cụm", "Khám phá phân phối, tương quan và các quyết định làm sạch dữ liệu.")
    a, b, c = st.columns(3)
    a.metric("Trung vị số đơn", f"{customers.Frequency.median():.0f}")
    b.metric("Khách chỉ mua một lần", f"{customers.Frequency.eq(1).mean():.1%}")
    c.metric("Quốc gia", str(customers.Country.nunique()))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.subheader("Phân phối đặc trưng")
        variable = st.selectbox("Đặc trưng RFM", RFM, index=2, key="eda_variable")
        logged = st.toggle("Biến đổi log1p", value=True, key="eda_logged")
        frame = pd.DataFrame({"Giá trị": np.log1p(customers[variable]) if logged else customers[variable]})
        chart(alt.Chart(frame).mark_bar(color="#168C7F", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Giá trị:Q", bin=alt.Bin(maxbins=45), title=f"log1p({variable})" if logged else variable),
            y=alt.Y("count():Q", title="Số khách hàng"), tooltip=[alt.Tooltip("count():Q", title="Khách hàng")],
        ).properties(height=320))
        st.caption("Bật/tắt log1p để quan sát tác động lên dữ liệu lệch phải. Ma trận phân cụm chính còn được StandardScaler chuẩn hóa sau bước log.")
    with right:
        st.subheader("Tương quan Spearman")
        corr = customers[RFM].corr(method="spearman").rename_axis("Biến A").reset_index().melt(id_vars="Biến A", var_name="Biến B", value_name="Tương quan")
        base = alt.Chart(corr).encode(x=alt.X("Biến A:N", title=None), y=alt.Y("Biến B:N", title=None))
        heat = base.mark_rect().encode(color=alt.Color("Tương quan:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")), tooltip=["Biến A:N", "Biến B:N", alt.Tooltip("Tương quan:Q", format=".3f")])
        text = base.mark_text(fontSize=17).encode(text=alt.Text("Tương quan:Q", format=".2f"), color=alt.condition("abs(datum['Tương quan']) > .6", alt.value("white"), alt.value("#172B4D")))
        chart((heat + text).properties(height=320))
    corr_fm = customers[RFM].corr(method="spearman").loc["Frequency", "Monetary"]
    takeaway(f"Frequency và Monetary có tương quan <b>{corr_fm:.3f}</b>: khách mua nhiều lần thường chi tiêu nhiều hơn. Đây là mối liên hệ quan sát, không phải kết luận nhân quả.")
    st.subheader("Từ giao dịch thô đến đặc trưng khách hàng")
    labels = {"Raw transaction rows": "Dòng giao dịch gốc", "Clean purchase rows": "Dòng mua hợp lệ sau ETL", "Duplicate purchase rows removed": "Dòng mua trùng được loại", "Product purchase rows": "Dòng mua sản phẩm", "Cancelled product rows": "Dòng hoàn/hủy xử lý riêng", "Gross RFM customers": "Khách RFM trước tinh chỉnh", "Customers before positive-net filter": "Khách sản phẩm trước lọc chi tiêu ròng", "Final customers": "Khách hàng cuối cùng"}
    counts = pd.DataFrame([(labels.get(k, k), v) for k, v in data["audit"]["counts"].items()], columns=["Chỉ tiêu", "Số lượng"])
    st.dataframe(counts, hide_index=True, width="stretch")
    st.caption("Các chỉ tiêu là những bước/đại lượng riêng; không cộng tất cả các dòng hoặc trừ lần lượt số dòng hoàn/hủy khỏi bảng mua.")
    with st.expander("Thống kê RFM và giả định xử lý"):
        st.dataframe(customers[RFM].describe().round(2), width="stretch")
        st.write("Loại giao dịch thiếu mã khách, lượng/giá không dương và hóa đơn hủy khỏi bảng mua; loại dòng trùng và mã ngoài quy tắc sản phẩm. Hoàn/hủy hợp lệ được tổng hợp riêng để điều chỉnh Monetary.")
        st.write("Giữ khách có chi tiêu ròng dương. Các giá trị RFM đồng hạng nhận cùng điểm baseline, nên số khách mỗi mức điểm không nhất thiết bằng nhau.")


def modeling(data):
    header("Modeling · So sánh và lựa chọn", "Một tập khách, cùng ba đặc trưng, cùng ma trận log1p + StandardScaler, không chặn ngoại lệ trong phép so sánh chính.")
    tab1, tab2, tab3 = st.tabs(["So sánh mô hình", "Chọn số cụm", "Độ ổn định và độ nhạy"])
    with tab1:
        comparison = data["comparison"]
        metric = st.selectbox("Chỉ số đánh giá", ["Silhouette", "DaviesBouldin", "CalinskiHarabasz"], key="model_metric")
        lower = metric == "DaviesBouldin"
        st.caption("Càng thấp càng tốt theo chỉ số này." if lower else "Càng cao càng tốt theo chỉ số này. Đây không phải accuracy.")
        chart(alt.Chart(comparison).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X(f"{metric}:Q", title=metric), y=alt.Y("Model:N", sort=alt.SortField(metric, order="ascending" if lower else "descending"), title=None),
            color=alt.condition(alt.datum.Model == "K-Means k=4", alt.value("#168C7F"), alt.value("#B4C4DD")),
            tooltip=["Model:N", "Components:Q", alt.Tooltip(f"{metric}:Q", format=".4f")],
        ).properties(height=300))
        st.dataframe(comparison.rename(columns={"Model": "Mô hình", "Components": "Số nhóm", "SmallestShare": "Tỷ lệ nhóm nhỏ nhất", "LargestShare": "Tỷ lệ nhóm lớn nhất"}).round(4), hide_index=True, width="stretch")
        takeaway("<b>K-Means k=4</b> là lựa chọn chính nhờ chất lượng phân cụm, độ ổn định và khả năng diễn giải. GMM k=4 có Davies–Bouldin tốt hơn; không có một mô hình thắng mọi tiêu chí.")
        st.info("So sánh trực tiếp K-Means, GMM và Ward ở cùng k=4. RFM scoring có 10 nhóm và GMM BIC có số nhóm riêng; cần cân nhắc độ chi tiết khi đọc metrics.")
    with tab2:
        variable = st.radio("Tiêu chí chọn k", ["Silhouette", "Inertia", "DaviesBouldin"], horizontal=True, key="k_metric")
        base = alt.Chart(data["selection"]).encode(x=alt.X("k:O", title="Số cụm k"), y=alt.Y(f"{variable}:Q", scale=alt.Scale(zero=False)), tooltip=["k:Q", alt.Tooltip(f"{variable}:Q", format=".4f")])
        curve = base.mark_line(point=True, color="#168C7F")
        selected = base.transform_filter(alt.datum.k == 4).mark_point(size=180, filled=True, color="#DA8B36")
        chart((curve + selected).properties(height=340))
        st.write("**Vì sao chọn k=4?** k=2 có silhouette cao hơn, nhưng mô tả hành vi còn thô. k=4 cân bằng mức chi tiết và khả năng đề xuất bốn hướng chăm sóc. Đây là quyết định có yếu tố nghiệp vụ, không phải tối ưu tuyệt đối về mọi chỉ số.")
        st.image(str(ROOT / "outputs/figures/ward_dendrogram.png"), caption="Ward: cấu trúc gộp nhóm và ngưỡng cắt bốn cụm.")
    with tab3:
        stability = data["stability"].groupby("Model").ARI.agg(["min", "median"]).reset_index().rename(columns={"Model": "Mô hình", "min": "ARI thấp nhất", "median": "ARI trung vị"})
        st.subheader("Thay đổi khởi tạo, kết quả có giữ được không?")
        st.dataframe(stability.round(4), hide_index=True, width="stretch")
        st.caption("Mọi cặp trong năm seed 0–4. ARI gần 1 nghĩa là phân nhóm gần giống nhau, kể cả khi hoán vị số hiệu nhãn.")
        st.subheader("GMM BIC nhạy với regularization")
        st.dataframe(data["gmm_sensitivity"].round(4), hide_index=True, width="stretch")
        st.write("BIC chọn cấu hình mô hình mật độ trong dải ứng viên. Không so trực tiếp BIC với inertia K-Means. Thay đổi regularization có thể thay số nhóm, nên GMM BIC được giữ làm phân tích khám phá.")
        st.subheader("Chặn ngoại lệ ở phân vị 99%")
        st.dataframe(data["sensitivity"].round(4), hide_index=True, width="stretch")
        st.caption("Mỗi hàng tính metrics trên ma trận tiền xử lý tương ứng. ARI so với bản không chặn đo mức thay đổi phân nhóm; không gộp các hàng thành xếp hạng trên cùng ma trận.")


def scenario(data):
    st.subheader("Thử tình huống RFM")
    st.write("Nhập một hồ sơ cùng định nghĩa RFM để xem K-Means đã huấn luyện gán vào nhóm nào. App sử dụng model đã lưu, không huấn luyện lại.")
    with st.form("scenario_form"):
        a, b, c = st.columns(3)
        r = a.number_input("Recency · ngày", min_value=1, max_value=730, value=8, step=1, key="scenario_r")
        f = b.number_input("Frequency · số đơn", min_value=1, max_value=500, value=10, step=1, key="scenario_f")
        m = c.number_input("Monetary · GBP", min_value=0.01, max_value=1000000., value=3622.95, step=50., format="%.2f", key="scenario_m")
        submitted = st.form_submit_button("Xem phân khúc", type="primary")
    if submitted:
        try:
            stamp = tuple((ROOT / "outputs/model" / name).stat().st_mtime_ns for name in ["kmeans.joblib", "gmm_k4.joblib"])
            result = predict_scenario([r, f, m], cached_models(stamp))
            km = cached_models(stamp)[0]
            if not np.array_equal(km["features"], RFM):
                raise ValueError("Model không sử dụng đúng ba đặc trưng RFM của demo.")
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(f"Không thể đọc model: {error}. Chạy lại python src/pipeline.py.")
            return
        st.success(f"Phân khúc K-Means: {result['segment']}")
        objective, action, kpi = ACTIONS[result["segment"]]
        st.markdown(f"**{objective}.** {action}\n\n**KPI gợi ý:** {kpi}.")
        customers = data["customers"]
        if any(v < customers[col].min() or v > customers[col].max() for col, v in zip(RFM, [r, f, m])):
            st.warning("Có giá trị ngoài khoảng dữ liệu huấn luyện. Kết quả là phép ngoại suy và cần được diễn giải thận trọng.")
        with st.expander("Xem xác suất từ GMM k=4"):
            chart(alt.Chart(result["probabilities"]).mark_bar(color="#A16BD8").encode(x="Thành phần:N", y=alt.Y("Xác suất:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")), tooltip=["Thành phần:N", alt.Tooltip("Xác suất:Q", format=".1%")]).properties(height=220))
            st.caption("Các thành phần GMM có hệ nhãn riêng. Xác suất này không phải độ chắc chắn của nhãn K-Means hay hiệu quả marketing.")
    st.caption("Tình huống minh họa trên cùng cửa sổ quan sát và định nghĩa RFM; chưa phải dự báo hành vi tương lai.")


def visualize(data):
    customers, profiles = data["customers"], data["profiles"]
    header("Visualize · Khám phá bốn nhóm khách", "Lọc phân khúc, xem vị trí trên PCA, tìm khách hàng và thử một tình huống RFM.")
    left, right = st.columns([2, 1])
    segments = left.multiselect("Phân khúc hiển thị", list(COLORS), default=list(COLORS), key="segment_filter")
    country = right.selectbox("Quốc gia", ["Tất cả", *sorted(customers.Country.unique())], key="country_filter")
    selected = filter_customers(customers, segments, country)
    st.caption(f"Bộ lọc hiện tại: {len(selected):,} / {len(customers):,} khách. Các chỉ số bên dưới được tính lại theo bộ lọc.")
    if selected.empty:
        st.info("Không có khách phù hợp. Hãy chọn ít nhất một phân khúc hoặc đổi quốc gia.")
    else:
        a, b, c = st.columns(3)
        a.metric("Khách sau lọc", f"{len(selected):,}")
        b.metric("Chi tiêu sau lọc", f"£{selected.Monetary.sum():,.0f}")
        c.metric("Trung vị Recency", f"{selected.Recency.median():.0f} ngày")
        variance = data["pca_variance"]
        chart(alt.Chart(selected).mark_circle(size=45, opacity=.6).encode(
            x=alt.X("PC1:Q", title=f"PC1 · {variance[0]:.1%} phương sai"),
            y=alt.Y("PC2:Q", title=f"PC2 · {variance[1]:.1%} phương sai"), color=color_encoding(),
            tooltip=["CustomerID:N", "Phân khúc:N", "Country:N", "Recency:Q", "Frequency:Q", alt.Tooltip("Monetary:Q", format=",.2f")],
        ).properties(height=430).interactive())
        st.caption("Rê chuột để xem khách, cuộn để phóng to và kéo để di chuyển. PCA là phép chiếu cùng ma trận RFM xuống 2D; model phân cụm dùng đủ ba đặc trưng. Màu là nhãn K-Means.")
        table = selected[["CustomerID", "Phân khúc", "Country", *RFM, "GMM_Confidence"]].copy()
        query = st.text_input("Tìm CustomerID trong bộ lọc hiện tại", placeholder="Ví dụ: 12347", key="customer_search")
        if query.strip():
            table = table[table.CustomerID.str.contains(query.strip(), regex=False)]
        st.dataframe(table.round(3), hide_index=True, width="stretch", height=260)
        st.download_button("Tải CSV đang hiển thị", table.to_csv(index=False).encode("utf-8-sig"), "khach_hang_da_loc.csv", "text/csv", key="download_customers")
    st.divider()
    st.subheader("Chân dung phân khúc · toàn bộ tập dữ liệu")
    segment = st.selectbox("Xem chân dung nhóm", list(COLORS), key="profile_segment")
    row = profiles.loc[profiles["Phân khúc"].eq(segment)].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Khách trong nhóm", f"{int(row.CustomerCount):,}", f"{row.CustomerShare:.1%} tập khách", delta_color="off")
    b.metric("Recency trung vị", f"{row.RecencyMedian:.0f} ngày")
    c.metric("Frequency trung vị", f"{row.FrequencyMedian:.0f} đơn")
    d.metric("Monetary trung vị", f"£{row.MonetaryMedian:,.2f}")
    objective, action, kpi = ACTIONS[segment]
    takeaway(f"<b>{objective}.</b> {action}<br><b>KPI:</b> {kpi}.")
    st.caption("Chân dung dùng toàn bộ tập dữ liệu, độc lập bộ lọc khám phá phía trên. Khách mới là tên tương đối: không phải mọi khách đều mới mua lần đầu. Đề xuất cần thử nghiệm có đối chứng.")
    st.divider()
    scenario(data)


def main():
    try:
        paths = [ROOT / "data/customer_features.csv", *sorted((ROOT / "outputs").rglob("*.csv")), *sorted((ROOT / "outputs").rglob("*.json"))]
        stamp = tuple((str(p.relative_to(ROOT)), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists())
        data = cached_data(stamp)
    except (FileNotFoundError, ValueError, KeyError, AssertionError) as error:
        st.error(f"Chưa sẵn sàng để hiển thị demo: {error}")
        st.code("python src/pipeline.py", language="bash")
        st.stop()
    with st.sidebar:
        st.markdown("## Retail Insight")
        st.caption("PHÂN KHÚC KHÁCH HÀNG")
        st.divider()
        page = st.radio("Nội dung trình bày", ["Giới thiệu", "EDA", "Modeling", "Visualize"], key="demo_page")
        st.divider()
        st.markdown("**Online Retail**")
        st.caption(f"{data['audit']['first_transaction'][:10]} — {data['audit']['last_transaction'][:10]}")
        st.caption(f"{len(data['customers']):,} khách · RFM · K-Means k=4")
        st.info("Demo dùng kết quả đã lưu. Không cần chạy lại huấn luyện khi đổi trang hoặc bộ lọc.")
    {"Giới thiệu": overview, "EDA": eda, "Modeling": modeling, "Visualize": visualize}[page](data)
    st.divider()
    st.caption("Bài tập Data Mining · Phân khúc khách hàng bằng RFM và phân cụm · Các kết luận mô tả dữ liệu lịch sử.")


if __name__ == "__main__":
    main()
