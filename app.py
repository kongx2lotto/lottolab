"""
🎱 LottoLab 확률 분석 엔진 v5.0
AC값 · 빈도 가중치 · 구간 균형 · 17종 복합 필터링 알고리즘 기반 Monte Carlo 시뮬레이션
"""
import streamlit as st
import pandas as pd
import requests, random, time, json, os
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from itertools import combinations

st.set_page_config(page_title="🎱 LottoLab v5.0", page_icon="🎱", layout="wide")

CACHE_FILE = "lotto_cache.json"

# ============================================================
# 1. 하이브리드 데이터 로드
# ============================================================
def load_from_excel():
    try:
        df = pd.read_excel("lotto.xlsx", engine="openpyxl")
    except FileNotFoundError:
        st.error("❌ lotto.xlsx 파일이 없습니다! superkts.com/lotto/download 에서 다운받아 같은 폴더에 넣어주세요.")
        return []
    data = []
    for _, row in df.iterrows():
        try:
            cols = list(row.values)
            round_no = int(cols[0])
            date_val = str(cols[1])
            nums = sorted([int(cols[i]) for i in range(1, 7)])
            bonus = int(cols[7])
            data.append({"round": round_no, "date": date_val, "numbers": nums, "bonus": bonus})
        except (ValueError, IndexError, TypeError):
            continue
    data.sort(key=lambda x: x["round"])
    return data

def fetch_from_api(start_round):
    new_data = []
    current = start_round
    consecutive_fails = 0
    while consecutive_fails < 3:
        try:
            url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={current}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            result = res.json()
            if result.get("returnValue") == "success":
                nums = sorted([result[f"drwtNo{j}"] for j in range(1, 7)])
                new_data.append({
                    "round": result["drwNo"],
                    "date": result["drwNoDate"],
                    "numbers": nums,
                    "bonus": result["bnusNo"]
                })
                current += 1
                consecutive_fails = 0
                time.sleep(0.2)
            else:
                consecutive_fails += 1
                current += 1
        except:
            consecutive_fails += 1
            current += 1
    return new_data

def save_cache(data):
    cache = []
    for d in data:
        cache.append({
            "round": d["round"],
            "date": d["date"],
            "numbers": d["numbers"],
            "bonus": d["bonus"]
        })
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@st.cache_data(ttl=3600)
def load_all_data():
    cached = load_cache()
    if cached:
        max_cached = max(d["round"] for d in cached)
        new_data = fetch_from_api(max_cached + 1)
        if new_data:
            all_data = cached + new_data
            all_data.sort(key=lambda x: x["round"])
            seen = set()
            unique = []
            for d in all_data:
                if d["round"] not in seen:
                    seen.add(d["round"])
                    unique.append(d)
            save_cache(unique)
            return unique
        return cached

    excel_data = load_from_excel()
    if not excel_data:
        return []

    max_excel = max(d["round"] for d in excel_data)
    st.info(f"📂 엑셀에서 {len(excel_data)}회차 로드 완료 (1~{max_excel}회)")

    new_data = fetch_from_api(max_excel + 1)
    if new_data:
        st.info(f"🌐 API에서 {len(new_data)}회차 추가 ({max_excel+1}~{max_excel+len(new_data)}회)")

    all_data = excel_data + new_data
    all_data.sort(key=lambda x: x["round"])
    save_cache(all_data)
    return all_data

# ============================================================
# 2. 분석 함수들
# ============================================================
def calc_ac(nums):
    diffs = set()
    for a, b in combinations(nums, 2):
        diffs.add(abs(a - b))
    return len(diffs) - 5

def calc_odd_even(nums):
    odds = sum(1 for n in nums if n % 2)
    return odds, 6 - odds

def calc_high_low(nums):
    low = sum(1 for n in nums if n <= 22)
    return low, 6 - low

def calc_consecutive(nums):
    s = sorted(nums)
    max_c = cur = 1
    for i in range(1, len(s)):
        if s[i] - s[i-1] == 1:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 1
    return max_c

def calc_section_dist(nums):
    sec = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for n in nums:
        if n <= 10: sec[1] += 1
        elif n <= 20: sec[2] += 1
        elif n <= 30: sec[3] += 1
        elif n <= 40: sec[4] += 1
        else: sec[5] += 1
    return sec

def count_primes(nums):
    return sum(1 for n in nums if n in {2,3,5,7,11,13,17,19,23,29,31,37,41,43})

def count_perfect_squares(nums):
    return sum(1 for n in nums if n in {1,4,9,16,25,36})

def count_multiples_of_3(nums):
    return sum(1 for n in nums if n % 3 == 0)

def count_multiples_of_5(nums):
    return sum(1 for n in nums if n % 5 == 0)

def count_doubles(nums):
    return sum(1 for n in nums if n in {11, 22, 33, 44})

# ============================================================
# 3. 17가지 필터
# ============================================================
def passes_all_filters(nums, min_sum=100, max_sum=175, min_ac=7):
    total = sum(nums)
    if not (min_sum <= total <= max_sum): return False
    if calc_ac(nums) < min_ac: return False
    odds, evens = calc_odd_even(nums)
    if odds == 0 or odds == 6: return False
    low, high = calc_high_low(nums)
    if low == 0 or low == 6: return False
    last_digits = [n % 10 for n in nums]
    if max(Counter(last_digits).values()) >= 4: return False
    if not (15 <= sum(last_digits) <= 38): return False
    if calc_consecutive(nums) >= 3: return False
    if count_primes(nums) >= 4: return False
    if count_perfect_squares(nums) >= 3: return False
    if count_multiples_of_3(nums) >= 4: return False
    if count_multiples_of_5(nums) >= 3: return False
    if count_doubles(nums) >= 3: return False
    if nums[0] >= 15: return False
    if nums[-1] <= 30: return False
    sec = calc_section_dist(nums)
    if max(sec.values()) >= 4: return False
    return True

# ============================================================
# 4. 후보 풀 & 추천 생성
# ============================================================
def build_candidate_pool(data, pool_size=25):
    all_nums = [n for d in data for n in d["numbers"]]
    freq_all = Counter(all_nums)
    recent_nums = [n for d in data[-50:] for n in d["numbers"]]
    freq_recent = Counter(recent_nums)
    hot_nums = [n for d in data[-10:] for n in d["numbers"]]
    freq_hot = Counter(hot_nums)

    scores = {}
    for n in range(1, 46):
        scores[n] = freq_all.get(n,0)*1.0 + freq_recent.get(n,0)*3.0 + freq_hot.get(n,0)*5.0

    sections = {1: range(1,11), 2: range(11,21), 3: range(21,31), 4: range(31,41), 5: range(41,46)}
    size_per = {1: 5, 2: 6, 3: 6, 4: 5, 5: 3}

    pool = []
    for sid, rng in sections.items():
        top = sorted([(n, scores[n]) for n in rng], key=lambda x: x[1], reverse=True)[:size_per[sid]]
        pool.extend([n for n, _ in top])

    remaining = [n for n in range(1,46) if n not in pool]
    remaining.sort(key=lambda n: scores[n], reverse=True)
    while len(pool) < pool_size and remaining:
        pool.append(remaining.pop(0))
    pool.sort()
    return pool, scores

def generate_combinations(pool, scores, num_sets=5, min_sum=100, max_sum=175, min_ac=7):
    weights = [scores.get(n, 1) for n in pool]
    results = []
    attempts = 0
    while len(results) < num_sets and attempts < 500000:
        attempts += 1
        sel = set()
        while len(sel) < 6:
            sel.add(random.choices(pool, weights=weights, k=1)[0])
        nums = sorted(sel)
        if passes_all_filters(nums, min_sum, max_sum, min_ac) and nums not in results:
            results.append(nums)
    return results

# ============================================================
# 5. 백테스팅
# ============================================================
def run_backtest(data, test_rounds=100, games_per_round=10, pool_size=25):
    prize_table = {"5등":5000, "4등":50000, "3등":1500000, "2등":30000000, "1등":2000000000}
    results = {"1등":0, "2등":0, "3등":0, "4등":0, "5등":0, "꽝":0}
    total_cost = 0
    total_prize = 0
    start = max(100, len(data) - test_rounds)
    actual_rounds = len(data) - start
    progress = st.progress(0)

    for i in range(actual_rounds):
        idx = start + i
        past = data[:idx]
        actual = data[idx]
        actual_nums = set(actual["numbers"])
        bonus = actual["bonus"]
        pool, scores = build_candidate_pool(past, pool_size)
        weights = [scores.get(n,1) for n in pool]
        games = []
        att = 0
        while len(games) < games_per_round and att < 50000:
            att += 1
            sel = set()
            while len(sel) < 6:
                sel.add(random.choices(pool, weights=weights, k=1)[0])
            nums = sorted(sel)
            if passes_all_filters(nums) and nums not in games:
                games.append(nums)
        total_cost += len(games) * 1000
        for g in games:
            match = len(set(g) & actual_nums)
            bonus_match = bonus in g
            if match == 6: results["1등"] += 1; total_prize += prize_table["1등"]
            elif match == 5 and bonus_match: results["2등"] += 1; total_prize += prize_table["2등"]
            elif match == 5: results["3등"] += 1; total_prize += prize_table["3등"]
            elif match == 4: results["4등"] += 1; total_prize += prize_table["4등"]
            elif match == 3: results["5등"] += 1; total_prize += prize_table["5등"]
            else: results["꽝"] += 1
        progress.progress((i+1) / actual_rounds)

    progress.empty()
    return results, total_cost, total_prize, actual_rounds

# ============================================================
# 6. UI 유틸
# ============================================================
def ball_color(n):
    if n <= 10: return "#FBC400"
    elif n <= 20: return "#69C8F2"
    elif n <= 30: return "#FF7272"
    elif n <= 40: return "#AAAAAA"
    else: return "#B0D840"

def draw_balls(nums, bonus=None):
    cols = st.columns(len(nums) + (2 if bonus else 0))
    for i, n in enumerate(nums):
        color = ball_color(n)
        cols[i].markdown(
            f'<div style="background:{color};color:#000;border-radius:50%;'
            f'width:48px;height:48px;display:flex;align-items:center;'
            f'justify-content:center;font-weight:bold;font-size:18px;'
            f'margin:auto;">{n}</div>', unsafe_allow_html=True)
    if bonus:
        cols[len(nums)].markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:48px;font-size:24px;">+</div>', unsafe_allow_html=True)
        color = ball_color(bonus)
        cols[len(nums)+1].markdown(
            f'<div style="background:{color};color:#000;border-radius:50%;'
            f'width:48px;height:48px;display:flex;align-items:center;'
            f'justify-content:center;font-weight:bold;font-size:18px;'
            f'margin:auto;">{bonus}</div>', unsafe_allow_html=True)

# ============================================================
# 7. 메인 앱
# ============================================================
def main():
    st.title("🎱 LottoLab 확률 분석 엔진 v5.0")
    st.caption("AC값 · 빈도 가중치 · 구간 균형 · 17종 복합 필터링 알고리즘 기반 Monte Carlo 시뮬레이션")

    with st.spinner("📥 데이터 로딩 중..."):
        data = load_all_data()

    if not data:
        return

    latest = data[-1]
    st.success(f"✅ 1회 ~ {latest['round']}회 ({len(data)}회차) 데이터 로드 완료! (최신: {latest['date']})")

    st.markdown("**최신 당첨번호:**")
    draw_balls(latest["numbers"], latest["bonus"])
    st.markdown("---")

    menu = st.sidebar.radio("📋 메뉴", ["📊 통계 분석", "🎯 번호 추천", "🔬 백테스팅", "🔄 이월수 분석"])

    # ---- 📊 통계 분석 ----
    if menu == "📊 통계 분석":
        st.header("📊 통계 분석")
        recent_n = st.sidebar.slider("최근 N회차 분석", 50, len(data), 100)
        target = data[-recent_n:]

        st.subheader(f"최근 {recent_n}회차 번호 출현 빈도")
        all_nums = [n for d in target for n in d["numbers"]]
        freq = Counter(all_nums)
        freq_df = pd.DataFrame({"번호": list(range(1,46)),
                                 "출현횟수": [freq.get(i,0) for i in range(1,46)]})
        colors = [ball_color(n) for n in range(1,46)]
        fig = go.Figure(go.Bar(x=freq_df["번호"], y=freq_df["출현횟수"], marker_color=colors))
        fig.update_layout(title="번호별 출현 빈도", xaxis_title="번호",
                          yaxis_title="출현 횟수", xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("홀짝 비율 분포")
        oe_list = [calc_odd_even(d["numbers"]) for d in target]
        oe_counter = Counter(oe_list)
        oe_df = pd.DataFrame({"홀:짝": [f"{k[0]}:{k[1]}" for k in sorted(oe_counter.keys())],
                              "횟수": [oe_counter[k] for k in sorted(oe_counter.keys())]})
        fig2 = px.pie(oe_df, names="홀:짝", values="횟수", title="홀짝 비율 분포")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("AC값 분포")
        ac_list = [calc_ac(d["numbers"]) for d in target]
        ac_counter = Counter(ac_list)
        ac_df = pd.DataFrame({"AC값": sorted(ac_counter.keys()),
                               "횟수": [ac_counter[k] for k in sorted(ac_counter.keys())]})
        fig3 = go.Figure(go.Bar(x=ac_df["AC값"], y=ac_df["횟수"], marker_color="#69C8F2"))
        fig3.update_layout(title="AC값 분포", xaxis_title="AC값", yaxis_title="횟수")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("번호 합계 분포")
        sums = [sum(d["numbers"]) for d in target]
        fig4 = go.Figure(go.Histogram(x=sums, nbinsx=30, marker_color="#FF7272"))
        fig4.update_layout(title="번호 합계 히스토그램", xaxis_title="합계", yaxis_title="횟수")
        fig4.add_vrect(x0=100, x1=175, fillcolor="green", opacity=0.1,
                       annotation_text="100~175 구간")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("구간별 분포")
        sec_totals = {1:0, 2:0, 3:0, 4:0, 5:0}
        for d in target:
            sec = calc_section_dist(d["numbers"])
            for k in sec: sec_totals[k] += sec[k]
        sec_df = pd.DataFrame({"구간": ["1~10","11~20","21~30","31~40","41~45"],
                                "출현수": [sec_totals[i] for i in range(1,6)]})
        fig5 = go.Figure(go.Bar(x=sec_df["구간"], y=sec_df["출현수"],
                                marker_color=["#FBC400","#69C8F2","#FF7272","#AAAAAA","#B0D840"]))
        fig5.update_layout(title="구간별 번호 출현")
        st.plotly_chart(fig5, use_container_width=True)

    # ---- 🎯 번호 추천 ----
    elif menu == "🎯 번호 추천":
        st.header("🎯 번호 추천")
        num_sets = st.sidebar.slider("추천 세트 수", 1, 20, 5)
        pool, scores = build_candidate_pool(data)

        st.subheader(f"🏊 후보 번호 풀 ({len(pool)}개)")
        draw_balls(pool)
        st.markdown("---")

        if st.button("🎲 추천 번호 생성!", type="primary"):
            combos = generate_combinations(pool, scores, num_sets)
            if combos:
                for i, nums in enumerate(combos):
                    st.markdown(f"**세트 {i+1}**")
                    draw_balls(nums)
                    odds, evens = calc_odd_even(nums)
                    ac = calc_ac(nums)
                    st.caption(f"합계: {sum(nums)} | AC: {ac} | 홀:짝 {odds}:{evens} | "
                              f"연번: {calc_consecutive(nums)} | 끝수합: {sum(n%10 for n in nums)}")
                    st.markdown("---")
            else:
                st.warning("조건을 만족하는 조합을 찾지 못했습니다.")

    # ---- 🔬 백테스팅 ----
    elif menu == "🔬 백테스팅":
        st.header("🔬 백테스팅 (과거 데이터로 전략 검증)")
        test_rounds = st.sidebar.slider("테스트 회차 수", 50, 500, 100)
        games_per = st.sidebar.slider("회차당 게임 수", 5, 50, 10)

        if st.button("🚀 백테스팅 시작!", type="primary"):
            results, cost, prize, actual = run_backtest(data, test_rounds, games_per)
            total_games = actual * games_per

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 게임 수", f"{total_games:,}게임")
            col2.metric("총 투자금", f"{cost:,.0f}원")
            col3.metric("총 당첨금", f"{prize:,.0f}원")
            roi = (prize / cost * 100) if cost > 0 else 0
            col4.metric("ROI", f"{roi:.1f}%", delta=f"{roi-100:.1f}%")

            st.subheader("등수별 당첨 횟수")
            for rank in ["1등","2등","3등","4등","5등","꽝"]:
                cnt = results[rank]
                pct = cnt / total_games * 100 if total_games > 0 else 0
                st.write(f"**{rank}**: {cnt}회 ({pct:.2f}%)")

            res_df = pd.DataFrame({"등수": [k for k in results if k != "꽝"],
                                    "횟수": [results[k] for k in results if k != "꽝"]})
            if res_df["횟수"].sum() > 0:
                fig = px.bar(res_df, x="등수", y="횟수", title="등수별 당첨 분포", color="등수")
                st.plotly_chart(fig, use_container_width=True)

    # ---- 🔄 이월수 분석 ----
    elif menu == "🔄 이월수 분석":
        st.header("🔄 이월수 분석")
        st.markdown("직전 회차 번호 중 다음 회차에도 등장하는 '이월수' 패턴을 분석합니다.")

        carry_counts = []
        for i in range(1, len(data)):
            prev = set(data[i-1]["numbers"])
            curr = set(data[i]["numbers"])
            carry_counts.append(len(prev & curr))

        carry_counter = Counter(carry_counts)
        carry_df = pd.DataFrame({"이월수 개수": sorted(carry_counter.keys()),
                                  "횟수": [carry_counter[k] for k in sorted(carry_counter.keys())]})
        fig = go.Figure(go.Bar(x=carry_df["이월수 개수"], y=carry_df["횟수"],
                               marker_color="#B0D840"))
        fig.update_layout(title="이월수 개수 분포", xaxis_title="이월수 개수",
                          yaxis_title="횟수", xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        avg_carry = sum(carry_counts) / len(carry_counts)
        st.info(f"평균 이월수: **{avg_carry:.2f}개**")

        st.subheader(f"최신 {latest['round']}회 당첨번호")
        draw_balls(latest["numbers"], latest["bonus"])
        st.caption("이 번호들 중 1~2개가 다음 회차에도 나올 확률이 높습니다.")

    st.markdown("---")
    st.caption("⚠️ 이 프로그램은 재미와 데이터 분석 학습 목적입니다. "
               "로또는 완전한 독립 시행이며, 과거 데이터가 미래를 예측하지 않습니다.")

if __name__ == "__main__":
    main()
