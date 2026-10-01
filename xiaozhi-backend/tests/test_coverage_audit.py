from scripts.audit_cuisine_coverage import dish_belongs_to_scope, extract_dishes


def test_extracts_source_backed_dish_headings() -> None:
    content = """Bánh bèo bì

Bánh bèo bì được làm từ bột gạo, đậu xanh và bì heo, ăn kèm nước chấm chua ngọt.

Địa chỉ tham khảo:

Số 1 đường thử nghiệm.

Gỏi gà măng cụt

Gỏi gà măng cụt là món đặc sản có thịt gà, măng cụt và hương vị chua thanh.
"""

    assert extract_dishes("8 đặc sản Bình Dương", content) == [
        "Bánh bèo bì",
        "Gỏi gà măng cụt",
    ]


def test_title_candidate_is_conservative() -> None:
    assert extract_dishes(
        "Cách làm bún chả cá Nha Trang thơm ngon",
        "Bún được nấu từ cá và nước dùng đậm đà.",
    ) == ["bún chả cá Nha Trang"]
    assert extract_dishes(
        "Cách làm bánh tai Phú Thọ vỏ mềm dai nhân đậm vị ăn là mê",
        "Bánh được làm từ gạo và thịt.",
    ) == ["bánh tai Phú Thọ"]
    assert extract_dishes(
        "18 món đặc sản Huế nhất định phải thử",
        "Ẩm thực địa phương rất phong phú.",
    ) == []


def test_verified_dishes_and_mixed_region_titles_are_scoped() -> None:
    assert extract_dishes(
        "Ẩm thực thị xã — Phú Thọ",
        "Món được kiểm chứng: bánh tai, bánh cuốn\n\nBài viết mô tả nguyên liệu và hương vị.",
    ) == ["bánh tai", "bánh cuốn"]
    assert not dish_belongs_to_scope(
        "Bún chả cá Nha Trang", ["Nha Trang", "Hà Nội"], ["Hà Nội"]
    )
    assert dish_belongs_to_scope(
        "Bún chả cá Nha Trang", ["Nha Trang", "Hà Nội"], ["Khánh Hòa", "Nha Trang"]
    )
