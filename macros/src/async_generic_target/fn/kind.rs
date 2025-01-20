use proc_macro2::{Ident, Span};
use syn::{punctuated::Punctuated, FnArg, Generics, ReturnType, Token};

use crate::AsyncSignature;

pub struct Sync;

pub struct Async(pub(super) Option<AsyncSignature>);

pub trait Kind {
    fn asyncness() -> Option<Token![async]>;

    fn transform_ident(ident: Ident) -> Ident {
        ident
    }

    fn transform_constness(constness: Option<Token![const]>) -> Option<Token![const]> {
        constness
    }

    fn transform_generics(&mut self, generics: Generics) -> Generics {
        generics
    }

    fn transform_inputs(
        &mut self,
        inputs: Punctuated<FnArg, Token![,]>,
    ) -> Punctuated<FnArg, Token![,]> {
        inputs
    }

    fn transform_output(&mut self, output: ReturnType) -> ReturnType {
        output
    }
}

impl Kind for Sync {
    fn asyncness() -> Option<Token![async]> {
        None
    }
}

impl Kind for Async {
    fn asyncness() -> Option<Token![async]> {
        Some(Token![async](Span::call_site()))
    }

    fn transform_ident(ident: Ident) -> Ident {
        Ident::new(&format!("{ident}_async"), ident.span())
    }

    fn transform_generics(&mut self, generics: Generics) -> Generics {
        if let Some(alt_generics) = self
            .0
            .as_mut()
            .map(|AsyncSignature { generics, .. }| std::mem::take(generics))
        {
            alt_generics
        } else {
            generics
        }
    }

    fn transform_inputs(
        &mut self,
        inputs: Punctuated<FnArg, Token!(,)>,
    ) -> Punctuated<FnArg, Token!(,)> {
        if let Some(alt_inputs) = self
            .0
            .as_mut()
            .map(|AsyncSignature { inputs, .. }| std::mem::take(inputs))
        {
            alt_inputs
        } else {
            inputs
        }
    }

    fn transform_output(&mut self, output: ReturnType) -> ReturnType {
        if let Some(alt_output) = self.0.as_mut().map(|AsyncSignature { output, .. }| {
            let mut default = ReturnType::Default;
            std::mem::swap(output, &mut default);
            default
        }) {
            alt_output
        } else {
            output
        }
    }
}
