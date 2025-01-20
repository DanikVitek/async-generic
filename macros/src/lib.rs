#![deny(warnings)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg, doc_cfg_hide))]

use proc_macro::TokenStream;
use syn::parse_macro_input;

use self::async_generic_target::{r#fn::AsyncSignature, trait_part::LaterAttributes, TargetItem};

mod async_generic_target;
mod util;

#[proc_macro_attribute]
pub fn async_generic(args: TokenStream, input: TokenStream) -> TokenStream {
    let target_item: TargetItem = parse_macro_input!(input as TargetItem);

    match target_item {
        TargetItem::Fn(target_fn) => {
            let async_signature: Option<AsyncSignature> = if args.is_empty() {
                None
            } else {
                Some(parse_macro_input!(args as AsyncSignature))
            };
            async_generic_target::r#fn::expand(target_fn, async_signature).into()
        }
        TargetItem::TraitPart(target_trait_part) => {
            let later_attributes = if args.is_empty() {
                LaterAttributes::default()
            } else {
                parse_macro_input!(args as LaterAttributes)
            };
            async_generic_target::trait_part::expand(target_trait_part, later_attributes).into()
        }
    }
}
