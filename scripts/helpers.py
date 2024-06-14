def age_to_category(age, ranges):
    for category, (start, end) in ranges.items():
        if start <= age <= end:
            return category
    return "Unknown"
