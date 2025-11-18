from pipeline.step_6_compliance import check_rule_compliance
import json
import os

# Load rules from JSON file
RULES_PATH = "data/rules_p87.json"

def load_rules():
    if not os.path.exists(RULES_PATH):
        print(f"Rules file not found: {RULES_PATH}")
        return []
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    rules = load_rules()
    if not rules:
        return

    print(f"Starting Audit for {len(rules)} rules from {RULES_PATH}...\n")
    
    report = []
    
    for i, rule_obj in enumerate(rules):
        rule_text = rule_obj.get("text", "")
        rule_id = rule_obj.get("id", str(i+1))
        
        print(f"--- Rule {rule_id} ---")
        result = check_rule_compliance(rule_text)
        
        report_item = {
            "rule_id": rule_id,
            "section": rule_obj.get("section", ""),
            "rule_text": rule_text,
            "result": result
        }
        report.append(report_item)
        
        # Print brief result immediately
        status = result.get("status", "UNKNOWN")
        print(f"VERDICT: {status}")
        if status == "ВЫПОЛНЕНО":
            print(f"Evidence: {result.get('evidence')}")
            
    # Save full report
    with open("audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    print(f"\nAudit Complete. Report saved to 'audit_report.json'")

if __name__ == "__main__":
    main()

