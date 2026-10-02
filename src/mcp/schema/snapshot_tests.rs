use super::*;

#[test]
fn dump_snapshot_for_regeneration() {
    let actual = build_tools_list(&ModelManifest::default());
    let rendered = serde_json::to_string_pretty(&actual).unwrap();
    std::fs::write("/tmp/tools_list_new.json", rendered).unwrap();
}

#[test]
fn tools_list_matches_the_pre_split_snapshot() {
    let actual = build_tools_list(&ModelManifest::default());
    let expected: Value = serde_json::from_str(include_str!("tools_list.snapshot.json")).unwrap();
    assert_eq!(actual, expected);
}
