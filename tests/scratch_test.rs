#[test]
fn exact() {
    struct RawText<T>(T);
    impl<T: serde::Serialize> serde::Serialize for RawText<T> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            self.0.serialize(s)
        }
    }
    struct RawJson<'a>(&'a str);
    impl serde::Serialize for RawJson<'_> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_newtype_struct("$mini-swe-mcp:raw-json", &RawText(self.0))
        }
    }
    use serde::ser::SerializeStruct;
    struct Frame<'a>(&'a str);
    impl serde::Serialize for Frame<'_> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut st = s.serialize_struct("JsonRpcResponse", 4)?;
            st.serialize_field("jsonrpc", "2.0")?;
            st.serialize_field("id", &self.0.to_string())?;
            st.serialize_field("result", &serde_json::json!({}))?;
            st.end()
        }
    }
    struct Frame2<'a>(&'a str);
    impl serde::Serialize for Frame2<'_> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut st = s.serialize_struct("JsonRpcResponse", 4)?;
            st.serialize_field("jsonrpc", "2.0")?;
            st.serialize_field("id", &RawJson(self.0))?;
            st.serialize_field("result", &serde_json::json!({}))?;
            st.end()
        }
    }
    println!("(string) : {}", serde_json::to_string(&Frame("7")).unwrap());
    println!("(rawjson): {}", serde_json::to_string(&Frame2("7")).unwrap());
    let mut buf = Vec::new();
    {
        let mut ser = serde_json::Serializer::new(&mut buf);
        use serde::Serializer as _;
        Frame2("7").serialize(&mut ser).unwrap();
    }
    println!("(direct) : {}", String::from_utf8(buf).unwrap());
}
