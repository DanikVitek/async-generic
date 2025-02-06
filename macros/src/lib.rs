#![deny(warnings)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg, doc_cfg_hide))]

use proc_macro::TokenStream;
use syn::parse_macro_input;

use crate::async_generic_target::{r#fn, trait_part, TargetItem};

mod async_generic_target;
mod util;

#[cfg(test)]
pub(crate) mod test_helpers;

#[proc_macro_attribute]
pub fn async_generic(args: TokenStream, input: TokenStream) -> TokenStream {
    let target_item: TargetItem = parse_macro_input!(input as TargetItem);

    match target_item {
        TargetItem::Fn(target_fn) => {
            let args = parse_macro_input!(args as r#fn::AsyncGenericArgs);
            r#fn::expand(target_fn, args).into()
        }
        TargetItem::TraitPart(target_trait_part) => {
            let args = parse_macro_input!(args as trait_part::AsyncGenericArgs);
            trait_part::expand(target_trait_part, args).into()
        }
    }
}
