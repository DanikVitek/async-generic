pub mod cfg_attr;
pub mod cfg;

pub mod kw {
    use syn::custom_keyword;
    
    custom_keyword!(cfg);
    custom_keyword!(cfg_attr);
    
    pub mod predicate {
        use super::custom_keyword;

        custom_keyword!(all);
        custom_keyword!(any);
        custom_keyword!(not);
    }
}