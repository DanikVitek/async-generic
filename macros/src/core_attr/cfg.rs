use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use syn::{
    bracketed, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::{Bracket, Paren},
    AttrStyle, Attribute, LitStr, MacroDelimiter, Meta, MetaList, Path, PathSegment, Token,
};

use super::kw;

#[derive(Clone)]
pub struct CfgAttribute {
    pub pound_token: Token![#],
    pub style: AttrStyle,
    pub bracket_token: Bracket,
    pub cfg_token: kw::cfg,
    pub paren_token: Paren,
    pub meta: CfgMeta,
}

#[derive(Clone)]
pub struct CfgMeta {
    pub predicate: CfgPredicate,
}

impl From<CfgAttribute> for Attribute {
    fn from(value: CfgAttribute) -> Self {
        Self {
            pound_token: value.pound_token,
            style: value.style,
            bracket_token: value.bracket_token,
            meta: Meta::List(MetaList {
                path: Path::from(PathSegment::from(Ident::new("cfg", value.cfg_token.span))),
                delimiter: MacroDelimiter::Paren(value.paren_token),
                tokens: value.meta.into_token_stream(),
            }),
        }
    }
}

#[derive(Clone)]
pub enum CfgPredicate {
    Option(CfgOption),
    All(CfgAll),
    Any(CfgAny),
    Not(CfgNot),
}

#[derive(Clone)]
pub struct CfgOption {
    pub ident: Ident,
    pub value: Option<(Token![=], LitStr)>,
}

#[derive(Clone)]
pub struct CfgAll {
    pub all_token: kw::predicate::all,
    pub paren_token: Paren,
    pub predicates: Punctuated<CfgPredicate, Token![,]>,
}

#[derive(Clone)]
pub struct CfgAny {
    pub any_token: kw::predicate::any,
    pub paren_token: Paren,
    pub predicates: Punctuated<CfgPredicate, Token![,]>,
}

#[derive(Clone)]
pub struct CfgNot {
    pub not_token: kw::predicate::not,
    pub paren_token: Paren,
    pub predicate: Box<CfgPredicate>,
}

impl Parse for CfgAttribute {
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
            cfg_token: content1.parse()?,
            paren_token: parenthesized!(content2 in content1),
            meta: content2.parse()?,
        })
    }
}

impl ToTokens for CfgAttribute {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.pound_token.to_tokens(tokens);
        if let AttrStyle::Inner(not_token) = &self.style {
            not_token.to_tokens(tokens)
        }
        self.bracket_token.surround(tokens, |tokens| {
            self.cfg_token.to_tokens(tokens);
            self.paren_token.surround(tokens, |tokens| {
                self.meta.to_tokens(tokens);
            });
        });
    }
}

impl Parse for CfgMeta {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            predicate: input.parse()?,
        })
    }
}

impl ToTokens for CfgMeta {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.predicate.to_tokens(tokens);
    }
}

impl Parse for CfgPredicate {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let predicate = if input.peek(kw::predicate::all) {
            Self::All(input.parse()?)
        } else if input.peek(kw::predicate::any) {
            Self::Any(input.parse()?)
        } else if input.peek(kw::predicate::not) {
            Self::Not(input.parse()?)
        } else {
            Self::Option(input.parse()?)
        };
        Ok(predicate)
    }
}

impl ToTokens for CfgPredicate {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Option(value) => value.to_tokens(tokens),
            Self::All(value) => value.to_tokens(tokens),
            Self::Any(value) => value.to_tokens(tokens),
            Self::Not(value) => value.to_tokens(tokens),
        }
    }
}

impl Parse for CfgOption {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let value = if input.peek(Token![=]) {
            Some((input.parse()?, input.parse()?))
        } else {
            None
        };
        Ok(Self { ident, value })
    }
}

impl ToTokens for CfgOption {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        if let Some((eq, value)) = &self.value {
            eq.to_tokens(tokens);
            value.to_tokens(tokens);
        }
    }
}

impl Parse for CfgAll {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            all_token: input.parse()?,
            paren_token: parenthesized!(content in input),
            predicates: content.parse_terminated(CfgPredicate::parse, Token![,])?,
        })
    }
}

impl ToTokens for CfgAll {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.all_token.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.predicates.to_tokens(tokens);
        });
    }
}

impl Parse for CfgAny {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            any_token: input.parse()?,
            paren_token: parenthesized!(content in input),
            predicates: content.parse_terminated(CfgPredicate::parse, Token![,])?,
        })
    }
}

impl ToTokens for CfgAny {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.any_token.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.predicates.to_tokens(tokens);
        });
    }
}

impl Parse for CfgNot {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            not_token: input.parse()?,
            paren_token: parenthesized!(content in input),
            predicate: content.parse()?,
        })
    }
}

impl ToTokens for CfgNot {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.not_token.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.predicate.to_tokens(tokens);
        });
    }
}
