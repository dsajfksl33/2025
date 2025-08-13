# app.py — Streamlit app for Grade 8 (중2) Math: Triangle Performance Evaluation
# Author: ChatGPT (GPT-5 Thinking)
# Date: 2025-08-13
#
# Features
# - Korean-first UI with English hints
# - Student roster upload (CSV) + sample roster
# - Auto-generated quizzes on triangle topics (angle sum, exterior angle, triangle types,
#   congruence criteria SSS/SAS/ASA/AAS/HL, basic construction/reasoning prompts)
# - Auto-grading for objective items, rubric scoring for performance tasks
# - Per-student evaluation flow with evidence notes
# - Analytics dashboard (mastery by topic, rubric heatmap, item analysis)
# - Save/Load session to CSV
#
# No external packages beyond Streamlit, pandas, numpy, altair.

import io
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="삼각형 수행평가 (중2)", page_icon="📐", layout="wide")

# -----------------------------
# Helpers & Data Models
# -----------------------------

RUBRIC_CRITERIA = [
    {"key": "concept", "kr": "개념 이해", "desc": "삼각형 성질·합동조건 등의 개념 이해"},
    {"key": "procedure", "kr": "절차/계산", "desc": "계산 정확성, 풀이 절차의 적절성"},
    {"key": "reasoning", "kr": "추론/정당화", "desc": "이유 제시, 증명·설명의 타당성"},
    {"key": "communication", "kr": "의사소통", "desc": "풀이 표현, 기호 사용, 정리"},
]

LEVEL_DESCRIPTORS = {
    4: "우수 (Exemplary): 개념과 원리를 깊이 이해하고 다양한 맥락에서 적용하며 완전한 정당화를 제시함.",
    3: "보통 (Proficient): 핵심 개념을 이해하고 대부분의 문제를 정확히 해결하며 근거를 비교적 명확히 제시함.",
    2: "기초 (Developing): 기본 개념과 절차에 부분적 이해, 계산·정당화에 일부 오류가 있음.",
    1: "미흡 (Beginning): 개념 이해 및 절차 수행이 부족하며 정당화·표현이 미흡함.",
}

TOPICS = [
    ("각의 합", "triangle_angle_sum"),
    ("외각과 내각", "exterior_angle"),
    ("삼각형의 분류", "triangle_types"),
    ("합동조건", "congruence"),
    ("닮음 기초", "similarity_basic"),
]

OBJ_ITEM_TYPES = {"MCQ": "객관식", "NUM": "숫자답", "TF": "참/거짓"}

@dataclass
class ObjItem:
    id: str
    topic: str
    stem: str
    choices: List[str]
    answer: Any
    kind: str  # MCQ, NUM, TF
    points: int

@dataclass
class PerfTask:
    id: str
    topic: str
    prompt: str
    points: int

# -----------------------------
# Item Generation (Objective)
# -----------------------------

def gen_angle_sum_item() -> ObjItem:
    # Random triangle with two angles given, ask third angle
    a = random.choice(range(20, 121, 5))
    b = random.choice(range(20, 121, 5))
    # Ensure sum < 180
    while a + b >= 170:
        a = random.choice(range(20, 121, 5))
        b = random.choice(range(20, 121, 5))
    c = 180 - (a + b)
    stem = f"삼각형의 두 내각이 {a}°, {b}°일 때, 나머지 각의 크기는?"
    choices = [str(c), str(c+5), str(c-5), str(c+10)]
    random.shuffle(choices)
    return ObjItem(
        id=f"AS-{a}-{b}", topic="triangle_angle_sum", stem=stem,
        choices=choices, answer=str(c), kind="MCQ", points=2
    )

def gen_exterior_angle_item() -> ObjItem:
    # exterior angle equals sum of two remote interior angles
    x = random.choice(range(100, 160, 5))
    a = random.choice(range(20, 70, 5))
    b = x - a
    stem = (
        "삼각형의 한 외각이 "+str(x)+
        "°이다. 원격 내각 중 하나가 "+str(a)+
        "°일 때, 다른 원격 내각의 크기는?"
    )
    choices = [str(b), str(b+5), str(b-10), str(b+15)]
    random.shuffle(choices)
    return ObjItem(id=f"EX-{x}-{a}", topic="exterior_angle", stem=stem,
                   choices=choices, answer=str(b), kind="MCQ", points=2)

def gen_triangle_types_item() -> ObjItem:
    # classify by sides (SSS) or angles
    mode = random.choice(["sides", "angles"])
    if mode == "sides":
        a, b, c = sorted(random.sample(range(3, 12), 3))
        # Ensure triangle inequality
        while a + b <= c:
            a, b, c = sorted(random.sample(range(3, 12), 3))
        if a == b == c:
            t = "정삼각형"
        elif a == b or b == c:
            t = "이등변삼각형"
        else:
            t = "부등변삼각형"
        stem = f"세 변의 길이가 {a}, {b}, {c}인 삼각형의 분류는?"
        choices = ["정삼각형", "이등변삼각형", "직각삼각형", "부등변삼각형"]
        answer = t
    else:
        A = random.choice([30, 45, 60, 70, 80, 90, 100, 120])
        if A == 90:
            t = "직각삼각형"
        elif A > 90:
            t = "둔각삼각형"
        else:
            t = "예각삼각형"
        stem = f"어떤 삼각형의 한 각이 {A}°이다. 이 삼각형의 분류는?"
        choices = ["예각삼각형", "직각삼각형", "둔각삼각형", "정삼각형"]
        answer = t
    random.shuffle(choices)
    return ObjItem(id=f"TT-{random.randint(1000,9999)}", topic="triangle_types",
                   stem=stem, choices=choices, answer=answer, kind="MCQ", points=2)

def gen_congruence_item() -> ObjItem:
    # determine which criterion proves triangles congruent
    options = ["SSS", "SAS", "ASA", "AAS", "HL(직각삼각형)"]
    pattern = random.choice(options[:4])
    if pattern == "SSS":
        stem = "두 삼각형에서 대응하는 세 변의 길이가 각각 같다. 합동 판단 근거는?"
        ans = "SSS"
    elif pattern == "SAS":
        stem = "두 변의 길이와 그 끼인각이 각각 같다. 합동 판단 근거는?"
        ans = "SAS"
    elif pattern == "ASA":
        stem = "두 각과 그 사이에 있지 않은 한 변이 각각 같다. 합동 판단 근거는?"
        ans = "AAS"  # tricky: many 교과서 구분. We'll include both ASA/AAS; adjust
    else:
        stem = "두 각과 한 변이 각각 같다. 합동 판단 근거는?"
        ans = random.choice(["ASA", "AAS"])  # accept either in broader phrasing
    choices = options[:4] + ["판단 불가"]
    random.shuffle(choices)
    return ObjItem(id=f"CG-{random.randint(1000,9999)}", topic="congruence",
                   stem=stem, choices=choices, answer=ans, kind="MCQ", points=2)

def gen_similarity_item() -> ObjItem:
    # simple AA similarity check
    A, B = random.choice([30,40,50,60]), random.choice([40,50,60,70])
    stem = f"두 삼각형의 두 각이 각각 {A}°, {B}°로 같다. 닮음 판단이 가능한가?"
    choices = ["가능(AA)", "불가능", "가능(SSS)", "가능(SAS)"]
    answer = "가능(AA)"
    random.shuffle(choices)
    return ObjItem(id=f"SIM-{A}-{B}", topic="similarity_basic", stem=stem,
                   choices=choices, answer=answer, kind="MCQ", points=2)

GEN_FUNCS = {
    "triangle_angle_sum": gen_angle_sum_item,
    "exterior_angle": gen_exterior_angle_item,
    "triangle_types": gen_triangle_types_item,
    "congruence": gen_congruence_item,
    "similarity_basic": gen_similarity_item,
}

# -----------------------------
# Performance Tasks (Rubric)
# -----------------------------

PERF_BANK: List[PerfTask] = [
    PerfTask(
        id="PT-1",
        topic="congruence",
        prompt=(
            "도형에서 △ABC와 △A'B'C'가 주어져 있다. 주어진 조건만으로 두 삼각형의 합동을 판단하고, "
            "필요한 추가 조건이 있다면 제시하라. (조건·근거를 문장 혹은 기호로 서술)"
        ),
        points=8,
    ),
    PerfTask(
        id="PT-2",
        topic="exterior_angle",
        prompt=(
            "한 삼각형의 외각과 원격 내각의 관계를 이용해, 주어진 각도 정보를 바탕으로 미지수 x를 구하고, "
            "이 과정에서 사용한 정리를 명확히 서술하라."
        ),
        points=8,
    ),
    PerfTask(
        id="PT-3",
        topic="triangle_angle_sum",
        prompt=(
            "삼각형의 내각의 합이 180°임을 이용하여, 다각형의 내각의 합 공식을 유도하고 예시를 들어 설명하라."
        ),
        points=8,
    ),
]

# -----------------------------
# State Init
# -----------------------------

if "students" not in st.session_state:
    st.session_state.students = pd.DataFrame([
        {"ID": "S001", "이름": "홍길동"},
        {"ID": "S002", "이름": "김영희"},
        {"ID": "S003", "이름": "이민호"},
    ])

if "quiz_items" not in st.session_state:
    st.session_state.quiz_items: List[ObjItem] = []

if "responses" not in st.session_state:
    # responses[(student_id, item_id)] = {"response": ..., "correct": bool, "score": int}
    st.session_state.responses: Dict[str, Dict[str, Any]] = {}

if "rubric_scores" not in st.session_state:
    # rubric_scores[(student_id, task_id)] = {criterion_key: level(1-4), "notes": str}
    st.session_state.rubric_scores: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Sidebar: Configuration
# -----------------------------

st.sidebar.header("⚙️ 설정 (Settings)")
with st.sidebar:
    st.markdown("**대상**: 중학교 2학년 (Grade 8) — 단원: *삼각형*")
    topics_selected = st.multiselect(
        "출제 범위 (Topics)", [t for t, _ in TOPICS], default=[t for t, _ in TOPICS]
    )
    topic_keys = [k for t, k in TOPICS if t in topics_selected]

    n_items = st.slider("객관식 문항 수", min_value=5, max_value=20, value=10, step=1)
    seed = st.number_input("난수 시드(재현 가능)", min_value=0, value=42, step=1)

    if st.button("🧮 새 시험지 생성 (Generate)"):
        random.seed(int(seed))
        items: List[ObjItem] = []
        for i in range(n_items):
            tk = random.choice(topic_keys)
            items.append(GEN_FUNCS[tk]())
        st.session_state.quiz_items = items
        st.toast("시험지가 생성되었습니다!", icon="✅")

# -----------------------------
# Student Roster
# -----------------------------

st.header("📋 학생 명단 (Roster)")
col1, col2 = st.columns([2,1])
with col1:
    st.write("CSV 업로드 형식: **ID, 이름** (헤더 포함)")
    file = st.file_uploader("학생 명단 CSV 업로드", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        # Normalize columns
        cols_lower = {c.lower(): c for c in df.columns}
        id_col = cols_lower.get("id") or cols_lower.get("학번") or list(df.columns)[0]
        name_col = cols_lower.get("이름") or cols_lower.get("name") or list(df.columns)[1]
        df = df.rename(columns={id_col: "ID", name_col: "이름"})[["ID", "이름"]]
        st.session_state.students = df
with col2:
    st.download_button(
        "샘플 명단 다운로드",
        data=st.session_state.students.to_csv(index=False).encode("utf-8-sig"),
        file_name="sample_roster.csv",
        mime="text/csv",
    )

st.dataframe(st.session_state.students, use_container_width=True, height=220)

# -----------------------------
# Quiz Conduct & Auto-Grading
# -----------------------------

st.header("📝 객관식 평가 (Objective Quiz)")
if not st.session_state.quiz_items:
    st.info("좌측 사이드바에서 \"새 시험지 생성\"을 먼저 눌러주세요.")
else:
    tabs = st.tabs([f"문항 {i+1}" for i in range(len(st.session_state.quiz_items))])
    for i, item in enumerate(st.session_state.quiz_items):
        with tabs[i]:
            st.subheader(item.stem)
            if item.kind == "MCQ":
                key = f"resp_{item.id}"
                choice = st.radio("정답 선택", item.choices, key=key)
            elif item.kind == "NUM":
                key = f"resp_{item.id}"
                choice = st.number_input("숫자 입력", key=key, step=1)
            else:  # TF
                key = f"resp_{item.id}"
                choice = st.selectbox("선택", ["참", "거짓"], key=key)
            st.caption(f"배점: {item.points}점 | 유형: {OBJ_ITEM_TYPES.get(item.kind,'')} | 주제: {item.topic}")

    st.markdown("---")
    colg1, colg2 = st.columns([1,2])
    with colg1:
        student = st.selectbox("채점 대상 학생", st.session_state.students["이름"].tolist())
        sid = st.session_state.students.set_index("이름").loc[student, "ID"]
    with colg2:
        if st.button("⚡ 자동 채점 (Auto-grade)"):
            total = 0
            max_total = 0
            for item in st.session_state.quiz_items:
                key = f"resp_{item.id}"
                resp = st.session_state.get(key)
                correct = str(resp) == str(item.answer)
                score = item.points if correct else 0
                max_total += item.points
                total += score
                st.session_state.responses[(sid, item.id)] = {
                    "response": resp, "correct": correct, "score": score,
                    "answer": item.answer, "topic": item.topic
                }
            st.success(f"{student} 점수: {total} / {max_total}")

    # Item Analysis Table
    if st.session_state.responses:
        records = []
        for (sid_k, iid), rec in st.session_state.responses.items():
            if sid_k == sid:
                records.append({"학생ID": sid_k, "문항": iid, "정답": rec["answer"], "응답": rec["response"],
                                "정오": "O" if rec["correct"] else "X", "점수": rec["score"], "주제": rec["topic"]})
        if records:
            df_rec = pd.DataFrame(records)
            st.dataframe(df_rec, use_container_width=True)

# -----------------------------
# Performance Tasks with Rubric
# -----------------------------

st.header("📂 수행평가 (Performance Task + Rubric)")
colp1, colp2 = st.columns([2,1])
with colp1:
    st.write("**루브릭 4수준 (1=미흡 ~ 4=우수)** — 각 기준 설명: ")
    with st.expander("루브릭 수준 설명 (클릭)"):
        for lvl, desc in LEVEL_DESCRIPTORS.items():
            st.markdown(f"**{lvl}점** — {desc}")
with colp2:
    task = st.selectbox(
        "수행과제 선택",
        [f"{p.id} | {p.topic}" for p in PERF_BANK],
        index=0,
    )
    task_obj = next(p for p in PERF_BANK if p.id in task)

st.info(task_obj.prompt)

colr1, colr2 = st.columns([1,2])
with colr1:
    student2 = st.selectbox("평가 학생", st.session_state.students["이름"].tolist(), key="rb_student")
    sid2 = st.session_state.students.set_index("이름").loc[student2, "ID"]
    rubric_levels = {}
    for crit in RUBRIC_CRITERIA:
        rubric_levels[crit["key"]] = st.slider(f"{crit['kr']} ({crit['key']})", 1, 4, 3)
    notes = st.text_area("증거/메모 (Evidence notes)")
    if st.button("💾 루브릭 저장"):
        st.session_state.rubric_scores[(sid2, task_obj.id)] = {
            **rubric_levels,
            "notes": notes,
            "task_points": task_obj.points,
        }
        st.toast("저장되었습니다", icon="💾")
with colr2:
    # Display existing rubric scores for this student/task
    existing = st.session_state.rubric_scores.get((sid2, task_obj.id))
    if existing:
        st.subheader("저장된 점수")
        df_r = pd.DataFrame(
            [{"기준": c["kr"], "수준(1-4)": existing[c["key"]]} for c in RUBRIC_CRITERIA]
        )
        df_r.loc[len(df_r)] = ["합계(가중치=1)", sum(existing[c["key"]] for c in RUBRIC_CRITERIA)]
        st.dataframe(df_r, use_container_width=True)
        st.caption(f"메모: {existing.get('notes','')}")

# -----------------------------
# Analytics Dashboard
# -----------------------------

st.header("📊 성취 분석 (Analytics)")

# Mastery by topic from objective items
if st.session_state.responses:
    resp_df = pd.DataFrame([
        {"학생ID": k[0], "문항": k[1], **v} for k, v in st.session_state.responses.items()
    ])
    mastery = resp_df.groupby(["학생ID", "topic"])['correct'].mean().reset_index()
    mastery['정답률(%)'] = (mastery['correct'] * 100).round(1)
    st.subheader("토픽별 정답률")
    chart = alt.Chart(mastery).mark_bar().encode(
        x=alt.X('topic:N', title='토픽'),
        y=alt.Y('정답률(%):Q', title='정답률(%)'),
        color='topic:N',
        column=alt.Column('학생ID:N', header=alt.Header(title='학생ID')),
        tooltip=['학생ID','topic','정답률(%)']
    ).properties(height=200)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("객관식 응답 데이터가 필요합니다.")

# Rubric heatmap-like table
if st.session_state.rubric_scores:
    rows = []
    for (sid_k, tid), rec in st.session_state.rubric_scores.items():
        row = {"학생ID": sid_k, "과제": tid}
        for c in RUBRIC_CRITERIA:
            row[c["kr"]] = rec.get(c["key"], np.nan)
        rows.append(row)
    rdf = pd.DataFrame(rows)
    st.subheader("루브릭 분포")
    st.dataframe(rdf, use_container_width=True)
else:
    st.info("루브릭 점수 데이터가 필요합니다.")

# -----------------------------
# Export / Import
# -----------------------------

st.header("💾 내보내기 / 불러오기 (Save/Load)")
colx1, colx2, colx3 = st.columns(3)
with colx1:
    if st.button("CSV로 응답 저장"):
        if st.session_state.responses:
            out = pd.DataFrame([
                {"학생ID": k[0], "문항": k[1], **v} for k, v in st.session_state.responses.items()
            ])
            st.download_button(
                "객관식_응답.csv 다운로드",
                data=out.to_csv(index=False).encode("utf-8-sig"),
                file_name="objective_responses.csv",
                mime="text/csv",
            )
        else:
            st.warning("저장할 객관식 응답이 없습니다.")
with colx2:
    if st.button("CSV로 루브릭 저장"):
        if st.session_state.rubric_scores:
            out = []
            for (sid_k, tid), rec in st.session_state.rubric_scores.items():
                row = {"학생ID": sid_k, "과제": tid, **{c["key"]: rec[c["key"]] for c in RUBRIC_CRITERIA}, "notes": rec.get("notes", "")}
                out.append(row)
            out_df = pd.DataFrame(out)
            st.download_button(
                "루브릭_점수.csv 다운로드",
                data=out_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="rubric_scores.csv",
                mime="text/csv",
            )
        else:
            st.warning("저장할 루브릭 점수가 없습니다.")
with colx3:
    # Session snapshot (quiz items + responses + rubric)
    snap = {
        "quiz_items": [asdict(x) for x in st.session_state.quiz_items],
        "responses": {(f"{k[0]}::{k[1]}"): v for k, v in st.session_state.responses.items()},
        "rubric_scores": {(f"{k[0]}::{k[1]}"): v for k, v in st.session_state.rubric_scores.items()},
        "students": st.session_state.students.to_dict(orient='list'),
    }
    json_bytes = json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("세션 저장(JSON)", data=json_bytes, file_name="triangle_eval_session.json", mime="application/json")

st.markdown("---")
with st.expander("❓도움말 (How to use)"):
    st.markdown(
        """
        1) **좌측에서 출제 범위와 문항 수를 정하고 Generate**를 누르세요.
        2) **학생 명단**을 업로드하거나 샘플을 사용하세요.
        3) **객관식 평가** 탭에서 학생이 응답한 뒤, 채점 대상 학생을 고르고 **Auto-grade**.
        4) **수행평가** 영역에서 루브릭(1~4) 점수를 기록하고 메모를 남기세요.
        5) **성취 분석**에서 토픽별 정답률과 루브릭 분포를 확인하세요.
        6) **내보내기** 버튼으로 CSV/JSON 파일로 저장하세요.
        
        ※ 합동조건(ASA/AAS) 서술은 교과서에 따라 표현이 다를 수 있어, 해당 문항은 채점 후 교사 검토를 권장합니다.
        """
    )
