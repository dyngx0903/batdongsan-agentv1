#!/usr/bin/env python3
"""Test script to demonstrate improved persona descriptions."""

from agent.comparison_formatter import ComparisonFormatter

# Test case 1: Investor property vs. Family home
formatter = ComparisonFormatter()

# Listing A: Low price/m², large area → Investor persona
listing_a = {
    "price_value_vnd": 3_000_000_000,  # 3B VND
    "area_m2": 50,  # 50m²
    "bedrooms": 1,
    "bathrooms": 1,
    "district": "Bình Thạnh",
    "project": "Ascent Garden",
    "amenities_area": ["Quán café", "Siêu thị"],
    "nearby_transport": ["Metro gần"],
    "suitable_for": ["Nhà đầu tư"],
}

# Listing B: Family home with more space
listing_b = {
    "price_value_vnd": 4_500_000_000,  # 4.5B VND
    "area_m2": 120,  # 120m²
    "bedrooms": 3,
    "bathrooms": 2,
    "district": "Thủ Đức",
    "project": "Green Valley",
    "amenities_area": ["Trường học", "Công viên", "Siêu thị"],
    "nearby_transport": ["Metro", "Bus"],
    "suitable_for": ["Gia đình", "Ở lâu dài"],
}

print("=" * 80)
print("TEST: Improved Persona Descriptions")
print("=" * 80)

result = formatter.format_comparison(
    listing_a=listing_a,
    listing_b=listing_b,
    user_query="Tìm nhà để ở hoặc đầu tư",
)

# Extract and display only the persona section
lines = result.split("\n")
persona_start = None
for i, line in enumerate(lines):
    if "## 👥 Mức độ phù hợp" in line:
        persona_start = i
        break

if persona_start is not None:
    persona_section = "\n".join(lines[persona_start:persona_start + 5])
    print("\nPERSONA SECTION:")
    print(persona_section)
else:
    print("\nFull result (persona section):")
    print(result)

print("\n" + "=" * 80)
print("TEST 2: Single vs. Couple")
print("=" * 80)

listing_single = {
    "price_value_vnd": 2_000_000_000,
    "area_m2": 35,
    "bedrooms": 1,
    "bathrooms": 1,
    "district": "Bình Thạnh",
    "amenities_area": ["Café", "Gym", "Quán ăn"],
    "nearby_transport": ["Metro Bình Thạnh"],
    "suitable_for": ["Single", "Sinh viên"],
}

listing_couple = {
    "price_value_vnd": 2_800_000_000,
    "area_m2": 60,
    "bedrooms": 2,
    "bathrooms": 1,
    "district": "Bình Thạnh",
    "amenities_area": ["Quán cà phê", "Nhà hàng", "Không gian xanh"],
    "nearby_transport": ["Bus"],
    "suitable_for": ["Cặp đôi"],
}

result2 = formatter.format_comparison(
    listing_a=listing_single,
    listing_b=listing_couple,
    user_query="Tìm căn hộ gần metro",
)

lines2 = result2.split("\n")
for i, line in enumerate(lines2):
    if "## 👥 Mức độ phù hợp" in line:
        persona_section2 = "\n".join(lines2[i:i + 5])
        print("\nPERSONA SECTION (TEST 2):")
        print(persona_section2)
        break

print("\n✅ Tests completed!")
