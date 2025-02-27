use proc_macro2::Ident;
use syn::{punctuated::Punctuated, Attribute, FnArg, Generics, ReturnType, Token, Variadic};

use super::{AsyncSignature, SyncSignature};
use crate::async_generic_target::r#fn::InterfaceKind;

pub struct Sync(pub(super) Option<SyncSignature>);

pub struct Async<const PRESERVE_IDENT: bool>(pub(super) Option<AsyncSignature>);

pub trait Kind {
    fn transform_constness(constness: Option<Token![const]>) -> Option<Token![const]> {
        constness
    }

    fn asyncness(&self) -> Option<Token![async]>;

    fn extend_attrs(&mut self, attrs: Vec<Attribute>) -> Vec<Attribute>;

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

    fn transform_variadic(&mut self, variadic: Option<Variadic>) -> Option<Variadic> {
        variadic
    }

    fn transform_output(&mut self, output: ReturnType) -> ReturnType {
        output
    }
}

impl Kind for Sync {
    fn asyncness(&self) -> Option<Token![async]> {
        None
    }

    fn extend_attrs(&mut self, mut attrs: Vec<Attribute>) -> Vec<Attribute> {
        attrs.extend(self.0.take().into_iter().flat_map(|sig| sig.attrs));
        attrs
    }
}

impl<const PRESERVE_IDENT: bool> Kind for Async<PRESERVE_IDENT> {
    fn transform_constness(_constness: Option<Token![const]>) -> Option<Token![const]> {
        None // TODO: return `constness` when `const async fn` is stabilized
    }

    fn asyncness(&self) -> Option<Token![async]> {
        matches!(
            self.0,
            None | Some(AsyncSignature {
                interface_kind: InterfaceKind::AsyncFn,
                ..
            })
        )
        .then(<Token![async]>::default)
    }

    fn extend_attrs(&mut self, mut attrs: Vec<Attribute>) -> Vec<Attribute> {
        if let Some(alt_attrs) = self.0.as_mut().map(|sig| core::mem::take(&mut sig.attrs)) {
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
        if let Some(alt_generics) = self.0.as_mut().and_then(|sig| {
            sig.params
                .as_mut()
                .map(|params| core::mem::take(&mut params.generics))
        }) {
            alt_generics
        } else {
            generics
        }
    }

    fn transform_inputs(
        &mut self,
        inputs: Punctuated<FnArg, Token![,]>,
    ) -> Punctuated<FnArg, Token![,]> {
        if let Some(alt_inputs) = self.0.as_mut().and_then(|sig| {
            sig.params
                .as_mut()
                .map(|params| core::mem::take(&mut params.inputs))
        }) {
            alt_inputs
        } else {
            inputs
        }
    }

    fn transform_variadic(&mut self, variadic: Option<Variadic>) -> Option<Variadic> {
        if let Some(alt_variadic) = self
            .0
            .as_mut()
            .and_then(|sig| sig.params.as_mut().map(|params| params.variadic.take()))
        {
            alt_variadic
        } else {
            variadic
        }
    }

    fn transform_output(&mut self, output: ReturnType) -> ReturnType {
        if let Some(alt_output) = self.0.as_mut().and_then(|sig| {
            sig.params.as_mut().map(|params| {
                let mut default = ReturnType::Default;
                core::mem::swap(&mut params.output, &mut default);
                default
            })
        }) {
            alt_output
        } else {
            output
        }
    }
}
