"""Geographic search scopes for Vietnam's post-2025 provinces.

Dialogue keeps the place name spoken by the user. Retrieval may search a wider
scope containing former provinces and well-known destinations inside that
place. This prevents a query for a current province from hiding useful older
articles whose titles still use the former administrative name.
"""

from __future__ import annotations


# Members are search labels already present in dialogue.REGION_ALIASES. The
# first item is always the current province/city name shown to the user.
CURRENT_PROVINCE_MEMBERS: dict[str, tuple[str, ...]] = {
    "Hà Nội": ("Hà Nội",),
    "Huế": ("Huế",),
    "Lai Châu": ("Lai Châu",),
    "Điện Biên": ("Điện Biên",),
    "Sơn La": ("Sơn La", "Mộc Châu"),
    "Lạng Sơn": ("Lạng Sơn",),
    "Cao Bằng": ("Cao Bằng",),
    "Tuyên Quang": ("Tuyên Quang", "Hà Giang", "Đồng Văn"),
    "Lào Cai": ("Lào Cai", "Yên Bái", "Sa Pa"),
    "Thái Nguyên": ("Thái Nguyên", "Bắc Kạn"),
    "Phú Thọ": ("Phú Thọ", "Vĩnh Phúc", "Hòa Bình"),
    "Bắc Ninh": ("Bắc Ninh", "Bắc Giang"),
    "Hưng Yên": ("Hưng Yên", "Thái Bình"),
    "Hải Phòng": ("Hải Phòng", "Hải Dương", "Cát Bà"),
    "Ninh Bình": ("Ninh Bình", "Hà Nam", "Nam Định"),
    "Quảng Ninh": ("Quảng Ninh", "Hạ Long", "Cô Tô"),
    "Thanh Hóa": ("Thanh Hóa",),
    "Nghệ An": ("Nghệ An",),
    "Hà Tĩnh": ("Hà Tĩnh",),
    "Quảng Trị": ("Quảng Trị", "Quảng Bình"),
    "Đà Nẵng": ("Đà Nẵng", "Quảng Nam", "Hội An"),
    "Quảng Ngãi": ("Quảng Ngãi", "Kon Tum"),
    "Gia Lai": ("Gia Lai", "Bình Định", "Quy Nhơn"),
    "Đắk Lắk": ("Đắk Lắk", "Phú Yên", "Buôn Ma Thuột", "Tuy Hòa"),
    "Khánh Hòa": ("Khánh Hòa", "Ninh Thuận", "Nha Trang", "Cam Ranh", "Phan Rang", "Ninh Hòa"),
    "Lâm Đồng": ("Lâm Đồng", "Đắk Nông", "Bình Thuận", "Đà Lạt", "Phan Thiết"),
    "TP.HCM": ("TP.HCM", "Bình Dương", "Vũng Tàu", "Côn Đảo"),
    "Đồng Nai": ("Đồng Nai", "Bình Phước", "Biên Hòa"),
    "Tây Ninh": ("Tây Ninh", "Long An"),
    "Cần Thơ": ("Cần Thơ", "Sóc Trăng", "Hậu Giang"),
    "Vĩnh Long": ("Vĩnh Long", "Bến Tre", "Trà Vinh"),
    "Đồng Tháp": ("Đồng Tháp", "Tiền Giang", "Mỹ Tho", "Sa Đéc"),
    "Cà Mau": ("Cà Mau", "Bạc Liêu"),
    "An Giang": ("An Giang", "Kiên Giang", "Phú Quốc", "Châu Đốc", "Rạch Giá", "Hà Tiên"),
}


# Former provinces remain valid, precise search labels. Tourist destinations
# additionally include their province label so generic provincial articles are
# available when the destination itself has little material.
DESTINATION_SCOPES: dict[str, tuple[str, ...]] = {
    "Nha Trang": ("Nha Trang", "Khánh Hòa"),
    "Cam Ranh": ("Cam Ranh", "Khánh Hòa"),
    "Sa Pa": ("Sa Pa", "Lào Cai"),
    "Hội An": ("Hội An", "Quảng Nam"),
    "Quy Nhơn": ("Quy Nhơn", "Bình Định"),
    "Buôn Ma Thuột": ("Buôn Ma Thuột", "Đắk Lắk"),
    "Đà Lạt": ("Đà Lạt", "Lâm Đồng"),
    "Phan Thiết": ("Phan Thiết", "Bình Thuận"),
    "Phú Quốc": ("Phú Quốc", "Kiên Giang"),
    "Vũng Tàu": ("Vũng Tàu",),
    "Hạ Long": ("Hạ Long", "Quảng Ninh"),
    "Cô Tô": ("Cô Tô", "Quảng Ninh"),
    "Cát Bà": ("Cát Bà", "Hải Phòng"),
    "Mộc Châu": ("Mộc Châu", "Sơn La"),
    "Đồng Văn": ("Đồng Văn", "Hà Giang"),
    "Phan Rang": ("Phan Rang", "Ninh Thuận"),
    "Ninh Hòa": ("Ninh Hòa", "Khánh Hòa"),
    "Tuy Hòa": ("Tuy Hòa", "Phú Yên"),
    "Côn Đảo": ("Côn Đảo", "Vũng Tàu"),
    "Mỹ Tho": ("Mỹ Tho", "Tiền Giang"),
    "Sa Đéc": ("Sa Đéc", "Đồng Tháp"),
    "Châu Đốc": ("Châu Đốc", "An Giang"),
    "Rạch Giá": ("Rạch Giá", "Kiên Giang"),
    "Hà Tiên": ("Hà Tiên", "Kiên Giang"),
    "Biên Hòa": ("Biên Hòa", "Đồng Nai"),
}


def search_regions(region: str) -> tuple[str, ...]:
    """Return labels whose titled sources belong to a requested place."""
    if region in CURRENT_PROVINCE_MEMBERS:
        return CURRENT_PROVINCE_MEMBERS[region]
    if region in DESTINATION_SCOPES:
        return DESTINATION_SCOPES[region]
    return (region,)


def region_matches(requested: str, found: str) -> bool:
    return found in search_regions(requested)

