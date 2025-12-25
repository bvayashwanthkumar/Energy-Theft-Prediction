def risk_level(score):
    if score > 0.8:
        return "High Risk"
    elif score > 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"
