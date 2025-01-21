use proc_macro2::{Ident, Span};
use syn::{punctuated::Punctuated, Attribute, FnArg, Generics, ReturnType, Token};

use crate::AsyncSignature;

pub struct Sync;

pub struct Async<const PRESERVE_IDENT: bool> {
    pub(super) signature: Option<AsyncSignature>,
}

pub trait Kind {
    fn transform_constness(constness: Option<Token![const]>) -> Option<Token![const]> {
        constness
    }

    fn asyncness() -> Option<Token![async]>;

    fn extend_attrs(&mut self, attrs: Vec<Attribute>) -> Vec<Attribute> {
        attrs
    }

    fn transform_ident(ident: Ident) -> Ident {
        ident
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

impl<const PRESERVE_IDENT: bool> Kind for Async<PRESERVE_IDENT> {
    fn transform_constness(_constness: Option<Token!(const)>) -> Option<Token!(const)> {
        None // TODO: retutn `constness` when `const async fn` is stabilized
    }
    
    fn asyncness() -> Option<Token![async]> {
        Some(Token![async](Span::call_site()))
    }

    fn extend_attrs(&mut self, mut attrs: Vec<Attribute>) -> Vec<Attribute> {
        if let Some(alt_attrs) = self
            .signature
            .as_mut()
            .map(|AsyncSignature { attrs: attributes, .. }| std::mem::take(attributes))
        {
            attrs.extend(alt_attrs);
        }
        attrs
    }

    fn transform_ident(ident: Ident) -> Ident {
        if PRESERVE_IDENT {
            ident
        } else {
            Ident::new(&format!("{ident}_async"), ident.span())
        }
    }

    fn transform_generics(&mut self, generics: Generics) -> Generics {
        if let Some(alt_generics) = self
            .signature
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
            .signature
            .as_mut()
            .map(|AsyncSignature { inputs, .. }| std::mem::take(inputs))
        {
            alt_inputs
        } else {
            inputs
        }
    }

    fn transform_output(&mut self, output: ReturnType) -> ReturnType {
        if let Some(alt_output) = self.signature.as_mut().map(|AsyncSignature { output, .. }| {
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
