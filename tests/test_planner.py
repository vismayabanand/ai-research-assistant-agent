from src.planner import plan_reading

def test_plan_reading():
    papers = [
        {"title":"A","contributions":["c1","c2"],"gaps":["g1"],"comparisons":[]},
        {"title":"B","contributions":["c1"],"gaps":[],"comparisons":[]},
        {"title":"C","contributions":["c1","c2","c3"],"gaps":["g1","g2"],"comparisons":[]},
    ]
    ordered = plan_reading(papers)
    # C has 3 contributions, A has 2, B has 1
    assert [p["title"] for p in ordered] == ["C", "A", "B"]
