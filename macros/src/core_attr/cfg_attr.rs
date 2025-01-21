use proc_macro2::Ident;
use quote::ToTokens;
use syn::{
    bracketed, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::{Bracket, Paren},
    AttrStyle, Attribute, MacroDelimiter, Meta, MetaList, Path, PathSegment, Token,
};

use super::{cfg::CfgPredicate, kw};

#[derive(Clone)]
pub struct CfgAttrAttribute {
    pub pound_token: Token![#],
    pub style: AttrStyle,
    pub bracket_token: Bracket,
    pub cfg_attr_token: kw::cfg_attr,
    pub paren_token: Paren,
    pub meta: CfgAttrMeta,
}

#[derive(Clone)]
pub struct CfgAttrMeta {
    pub predicate: CfgPredicate,
    pub comma_token: Token![,],
    pub attrs: Punctuated<Meta, Token![,]>,
}

impl From<CfgAttrAttribute> for Attribute {
    fn from(value: CfgAttrAttribute) -> Self {
        Self {
            pound_token: value.pound_token,
            style: value.style,
            bracket_token: value.bracket_token,
            meta: Meta::List(MetaList {
                path: Path::from(PathSegment::from(Ident::new(
                    "cfg_attr",
                    value.cfg_attr_token.span,
                ))),
                delimiter: MacroDelimiter::Paren(value.paren_token),
                tokens: value.meta.into_token_stream(),
            }),
        }
    }
}

impl Parse for CfgAttrAttribute {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content1;
        let content2;
        Ok(Self {
            pound_token: input.parse()?,
            style: match input.parse::<Option<Token![!]>>() {
                Ok(Some(not_token)) => AttrStyle::Inner(not_token),
                Ok(None) => AttrStyle::Outer,
                Err(_) => unreachable!(),
            },
            bracket_token: bracketed!(content1 in input),
            cfg_attr_token: content1.parse()?,
            paren_token: parenthesized!(content2 in content1),
            meta: content2.parse()?,
        })
    }
}

impl ToTokens for CfgAttrAttribute {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.pound_token.to_tokens(tokens);
        if let AttrStyle::Inner(not_token) = &self.style {
            not_token.to_tokens(tokens);
        }
        self.bracket_token.surround(tokens, |tokens| {
            self.cfg_attr_token.to_tokens(tokens);
            self.paren_token.surround(tokens, |tokens| {
                self.meta.to_tokens(tokens);
            });
        });
    }
}

impl Parse for CfgAttrMeta {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            predicate: input.parse()?,
            comma_token: input.parse()?,
            attrs: input.parse_terminated(Meta::parse, Token![,])?,
        })
    }
}

impl ToTokens for CfgAttrMeta {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.predicate.to_tokens(tokens);
        self.comma_token.to_tokens(tokens);
        self.attrs.to_tokens(tokens);
    }
}
