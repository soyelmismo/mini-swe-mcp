use super::*;

#[test]
fn tools_list_matches_the_pre_split_snapshot() {
    let actual = build_tools_list(&ModelManifest::default());
    let expected: Value = serde_json::from_str(include_str!("tools_list.snapshot.json")).unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "snapshot regenerator"]
fn regenerate_snapshot() {
    let actual = build_tools_list(&ModelManifest::default());
    std::fs::write(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/mcp/schema/tools_list.snapshot.json"),
        serde_json::to_string_pretty(&actual).unwrap() + "\n",
    )
    .unwrap();
}
