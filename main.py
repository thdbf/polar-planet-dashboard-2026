import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

st.set_page_config(page_title="극지식물 생육 차이 분석 대시보드", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 데이터 읽기
# ============================================================

def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """utf-8-sig → cp949 순으로 읽어 인코딩 오류를 방지"""
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """학교마다 다른 열 이름을 표준 이름으로 통일"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


@st.cache_data
def load_env_df(data: bytes) -> pd.DataFrame:
    return std_cols(read_csv_bytes(data))


@st.cache_data
def load_growth_df(data: bytes, file_name: str) -> pd.DataFrame:
    df = read_csv_bytes(data)
    if not df.empty:
        return std_cols(df)
    if file_name and str(file_name).lower().endswith(".xlsx"):
        try:
            df = pd.read_excel(io.BytesIO(data))
            return std_cols(df)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ============================================================
# 공용 함수
# ============================================================

def get_school_name(env_file, growth_file):
    """학교명은 업로드한 파일명 앞부분(예: 인천아라고등학교_환경데이터.csv → 인천아라고등학교)에서 읽기(하드코딩 금지)"""
    for f in (env_file, growth_file):
        if f is None:
            continue
        base = os.path.splitext(f.name)[0]
        name = base.split("_")[0].strip()
        if name:
            return name
    return "아라고"


def fmt_id(v):
    """개체번호를 56.0처럼 소수점 없이 원래 형태 그대로 표시"""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)) and v.is_integer():
        return str(int(v))
    return str(v)


def download_button(df: pd.DataFrame, label: str, file_name: str):
    """한글 깨짐 방지 CSV 다운로드 버튼"""
    if df is None or df.empty:
        st.info("다운로드할 데이터가 없어요.")
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=file_name, mime="text/csv")


def growth_bar_chart(df: pd.DataFrame, item: str, top_n: int = 3):
    """개체별 평균 비교 막대 그래프 (상위 top_n 강조)"""
    tmp = df.dropna(subset=[item]).copy()
    if tmp.empty or tmp[item].nunique() < 2:
        st.info(f"⚠️ '{item}'은(는) 그래프로 그릴 데이터가 부족해요.")
        return
    g = tmp.groupby("개체번호")[item].mean().sort_values(ascending=False)
    top = g.head(top_n)
    rest = g.iloc[top_n:]
    colors = ["#1E90FF"] * len(g)
    colors[:top_n] = ["#FF8C00"] * top_n
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[fmt_id(v) for v in top.index], y=top.values,
        marker_color="#FF8C00", name="상위 3개체"))
    if len(rest):
        fig.add_trace(go.Bar(
            x=[fmt_id(v) for v in rest.index], y=rest.values,
            marker_color="#CCCCCC", name="기타"))
    fig.update_layout(
        title=f"개체별 평균 {item}", xaxis_title="개체번호", yaxis_title=item,
        height=400)
    st.plotly_chart(fig, width="stretch")


def scatter_with_trendline(x, y, xlab, ylab, title):
    """산점도 + np.polyfit 추세선 (px.scatter trendline 사용 금지 편의)"""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        st.info("⚠️ 산점도와 추세선을 그리려면 데이터가 3개 이상 필요해요.")
        return None
    x = tmp["x"].astype(float)
    y = tmp["y"].astype(float)
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        st.info("⚠️ 비교할 수 없는 데이터예요.")
        return None
    slope, intercept = np.polyfit(x, y, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="개체별",
        marker=dict(color="#1E90FF", size=8)))
    xs = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines", name="추세선",
        line=dict(color="#FF4500", width=2)))
    fig.update_layout(
        title=f"{title} (상관계수 r = {r:.3f})",
        xaxis_title=xlab, yaxis_title=ylab, height=400)
    return fig


def env_summary_table(df: pd.DataFrame):
    """있는 열만 골라 평균·최소·최대 표 만들기"""
    cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in df.columns]
    if not cols:
        return None
    s = df[cols].agg(["mean", "min", "max"]).T
    s = s.round(2)
    s.columns = ["평균", "최소", "최대"]
    return s


def main():
    school_name = get_school_name(env_file if "env_file" in dir() else None, None)
    st.title(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

    st.sidebar.title("📁 파일 업로드")
    env_file = st.sidebar.file_uploader(
        "환경 데이터 CSV", type=None,
        help="utf-8-sig 또는 cp949 인코딩 CSV")
    growth_file = st.sidebar.file_uploader(
        "생육 결과 데이터 CSV", type=None,
        help=".csv 또는 .xlsx")
    measure_items = st.sidebar.multiselect(
        "생육 측정 항목 선택",
        ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
        default=["생중량", "잎 수", "지상부 길이", "지하부 길이"])

    if env_file is None and growth_file is None:
        st.sidebar.info(
            "파일을 업로드하기 전에는 분석 결과를 볼 수 없어요. "
            "환경·생육 데이터를 하나씩 올려주세요.")

    env_df = load_env_df(env_file.getvalue()) if env_file is not None else pd.DataFrame()
    growth_df = load_growth_df(
        growth_file.getvalue(), growth_file.name) if growth_file is not None else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

    # ---------- Tab 1: 실험 개요 ----------
    with tab1:
        try:
            st.header("1. 연구 배경 및 목적")
            st.write(
                f"{school_name} 극지 식물은 생육 환경이 까다로워 같은 종이라도 개체마다 "
                "생육 차이가 커요. 이 대시보드는 환경 조건과 개체별 생육 데이터를 연결해, "
                "어느 개체가 잘 자랐는지와 환경 요인의 관계를 한눈에 보기 위한 교육용 대시보드예요.")
            st.header("2. 데이터 요약")
            rows = []
            if not growth_df.empty:
                rows.append(["학교명", school_name])
                rows.append(["생육 개체 수", f"{len(growth_df)}개체"])
            if not env_df.empty:
                if "time" in env_df.columns:
                    pmin = env_df["time"].min().strftime("%Y-%m-%d")
                    pmax = env_df["time"].max().strftime("%Y-%m-%d")
                    rows.append(["환경 측정 기간", f"{pmin} ~ {pmax}"])
                cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                rows.append(["환경 측정 항목", ", ".join(cols) if cols else "없음"])
            if not growth_df.empty:
                gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
                rows.append(["생육 측정 항목", ", ".join(gcols) if gcols else "없음"])
            st.table(pd.DataFrame(rows, columns=["항목", "내용"]))
            st.header("3. 주요 지표 카드")
            c1, c2, c3, c4 = st.columns(4)
            if "temperature" in env_df.columns and env_df["temperature"].notna().sum() >= 1:
                c1.metric("평균 온도", f"{env_df['temperature'].mean():.2f} °C")
            else:
                c1.metric("평균 온도", "데이터 없음")
            if "humidity" in env_df.columns and env_df["humidity"].notna().sum() >= 1:
                c2.metric("평균 습도", f"{env_df['humidity'].mean():.2f} %")
            else:
                c2.metric("평균 습도", "데이터 없음")
            if "생중량" in growth_df.columns and growth_df["생중량"].notna().sum() >= 1:
                c3.metric("평균 생중량", f"{growth_df['생중량'].mean():.2f} g")
            else:
                c3.metric("평균 생중량", "데이터 없음")
            c4.metric("총 개체수", f"{len(growth_df)}개체" if not growth_df.empty else "데이터 없음")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ---------- Tab 2: 환경 데이터 ----------
    with tab2:
        try:
            st.header("환경 데이터")
            if env_df.empty:
                st.info("환경 데이터가 아직 없어요. 사이드바에서 CSV를 업로드하면 표를 표시할 수 있어요.")
            else:
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                if not env_cols:
                    st.info("온도·습도·pH·EC 열이 하나도 없어요.")
                else:
                    st.subheader("환경 요약 (평균·최소·최대)")
                    summary = env_summary_table(env_df)
                    st.dataframe(summary, width="stretch")
                    st.subheader("날짜별 시계열")
                    daily = env_df.dropna(subset=["time"]).copy()
                    if daily.empty:
                        st.info("시간 정보가 없어 일별 평균 차트를 그릴 수 없어요.")
                    else:
                        daily["date"] = daily["time"].dt.date
                        daily_mean = daily.groupby("date")[env_cols].mean().reset_index()
                        daily_mean["date"] = pd.to_datetime(daily_mean["date"])
                        has_ph_ec = "ph" in env_cols and "ec" in env_cols
                        if has_ph_ec:
                            fig = make_subplots(
                                rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("일별 평균 온도", "일별 평균 습도", "일별 평균 pH·EC"))
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["temperature"],
                                mode="lines", name="온도", line=dict(color="#1E90FF")), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["humidity"],
                                mode="lines", name="습도", line=dict(color="#228B22")), row=2, col=1)
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["ph"],
                                mode="lines", name="pH", line=dict(color="#8A2BE2")), row=3, col=1)
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["ec"],
                                mode="lines", name="EC", line=dict(color="#FF4500")), row=3, col=1)
                        else:
                            fig = make_subplots(
                                rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("일별 평균 온도", "일별 평균 습도"))
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["temperature"],
                                mode="lines", name="온도", line=dict(color="#1E90FF")), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=daily_mean["date"], y=daily_mean["humidity"],
                                mode="lines", name="습도", line=dict(color="#228B22")), row=2, col=1)
                        fig.update_layout(
                            title_text="환경 데이터 일별 변화",
                            height=650,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                        st.plotly_chart(fig, width="stretch")
            with st.expander("환경 데이터 원본 테이블"):
                st.dataframe(env_df, width="stretch")
                download_button(
                    env_df, f"{school_name} 환경 데이터 다운로드",
                    f"{school_name}_환경데이터.csv")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ---------- Tab 3: 생육 결과 ----------
    with tab3:
        try:
            st.header("생육 결과")
            if growth_df.empty:
                st.info("생육 데이터가 아직 없어요. 사이드바에서 CSV를 업로드하면 표를 표시할 수 있어요.")
            else:
                gw = growth_df["생중량"].dropna()
                if gw.empty:
                    st.info("생중량 데이터가 없어요.")
                else:
                    max_row = growth_df.loc[gw.idxmax()]
                    min_row = growth_df.loc[gw.idxmin()]
                    st.metric("평균 생중량", f"{gw.mean():.2f} g")
                    st.metric("최대 생중량 개체", f"{fmt_id(max_row['개체번호'])} ({gw.max():.2f} g)")
                    st.metric("최소 생중량 개체", f"{fmt_id(min_row['개체번호'])} ({gw.min():.2f} g)")
                    st.info(
                        f"전체 생중량: 평균 {gw.mean():.2f} g, "
                        f"표준편차 {gw.std():.2f} g, CV(변동계수) {gw.std() / gw.mean() * 100:.1f} %")
                    st.write(
                        "- 평균: 전체 개체 생중량의 평균, 표준편차: 개체별 값이 평균에서 퍼진 정도, "
                        "CV: 표준편차를 평균으로 나눈 값(%)으로 변동 크기를 비교")
                    st.write("---")
                    st.subheader("생중량 분포 (개체별)")
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=gw.values, name="생중량",
                        boxpoints="outliers", marker_color="#1E90FF"))
                    fig.update_layout(
                        title="생중량 분포", yaxis_title="생중량(g)", height=400)
                    st.plotly_chart(fig, width="stretch")
                    st.subheader("개체별 생육 데이터")
                    show_cols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
                    st.dataframe(growth_df[show_cols].dropna(), width="stretch")
                    st.write("---")
                    st.subheader("개체별 생육 비교")
                    for item in measure_items:
                        if item not in growth_df.columns:
                            st.warning(f"⚠️ '{item}' 열이 데이터에 없어요.")
                            continue
                        growth_bar_chart(growth_df, item)
                    st.write("---")
                    st.subheader("상관관계 분석")
                    fig1 = scatter_with_trendline(
                        growth_df["잎 수"], growth_df["생중량"],
                        "잎 수", "생중량(g)", "잎 수 vs 생중량")
                    fig2 = scatter_with_trendline(
                        growth_df["지상부 길이"], growth_df["생중량"],
                        "지상부 길이", "생중량(g)", "지상부 길이 vs 생중량")
                    if fig1 is not None:
                        st.plotly_chart(fig1, width="stretch")
                    if fig2 is not None:
                        st.plotly_chart(fig2, width="stretch")
                    st.write("---")
                    st.subheader("생중량 상위 25% vs 하위 25% 비교")
                    gw_mean = growth_df.groupby("개체번호")["생중량"].mean()
                    if len(gw_mean) >= 4:
                        q1 = gw_mean.quantile(0.25)
                        q3 = gw_mean.quantile(0.75)
                        low = gw_mean[gw_mean <= q1]
                        high = gw_mean[gw_mean >= q3]
                        st.write(f"하위 25% 기준: {q1:.2f} g 이하 | 평균 {low.mean():.2f} g")
                        st.write(f"상위 25% 기준: {q3:.2f} g 이상 | 평균 {high.mean():.2f} g")
                        st.write("하위 25% 개체: " + ", ".join(fmt_id(i) for i in low.index.tolist()))
                        st.write("상위 25% 개체: " + ", ".join(fmt_id(i) for i in high.index.tolist()))
                with st.expander("생육 데이터 원본 테이블"):
                    st.dataframe(growth_df, width="stretch")
                    download_button(
                        growth_df, f"{school_name} 생육 데이터 다운로드",
                        f"{school_name}_생육결과데이터.csv")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


if __name__ == "__main__":
    main()
