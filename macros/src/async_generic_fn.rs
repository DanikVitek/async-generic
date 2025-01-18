use std::borrow::Cow;

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{punctuated::Punctuated, visit_mut, Block, FnArg, Generics, ReturnType, Stmt, Token};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use crate::{
    util::{Either, LetExt},
    AsyncSignature, TargetFn,
};

pub struct AsyncGenericFn<'f, S> {
    target: &'f TargetFn,
    state: S,
}

impl<'f> AsyncGenericFn<'f, state::Sync> {
    pub const fn new(target: &'f TargetFn) -> Self {
        Self {
            target,
            state: state::Sync,
        }
    }
}

impl<'f> AsyncGenericFn<'f, state::Async> {
    pub const fn new(target: &'f TargetFn, sig: Option<AsyncSignature>) -> Self {
        Self {
            target,
            state: state::Async(sig),
        }
    }
}

pub mod state {
    use crate::AsyncSignature;

    pub struct Sync;

    pub struct Async(pub(super) Option<AsyncSignature>);
}

trait HasFunctionParts:
    HasConstness
    + HasUnsafety
    + HasAsyncness
    + HasIdent
    + HasInputs
    + HasGenerics
    + HasOutput
    + HasBlock
{
}

impl<T> HasFunctionParts for T where
    T: HasConstness
        + HasUnsafety
        + HasAsyncness
        + HasIdent
        + HasInputs
        + HasGenerics
        + HasOutput
        + HasBlock
{
}

trait HasConstness {
    fn constness(&self) -> Option<Token![const]>;
}

impl HasConstness for AsyncGenericFn<'_, state::Sync> {
    fn constness(&self) -> Option<Token![const]> {
        self.target.sig().constness
    }
}

impl HasConstness for AsyncGenericFn<'_, state::Async> {
    fn constness(&self) -> Option<Token![const]> {
        None // TODO: implement this when async functions are supported
    }
}

trait HasUnsafety {
    fn unsafety(&self) -> Option<Token![unsafe]>;
}

impl<S> HasUnsafety for AsyncGenericFn<'_, S> {
    fn unsafety(&self) -> Option<Token![unsafe]> {
        self.target.sig().unsafety
    }
}

trait HasAsyncness {
    fn asyncness(&self) -> Option<Token![async]>;
}

impl HasAsyncness for AsyncGenericFn<'_, state::Sync> {
    fn asyncness(&self) -> Option<Token![async]> {
        None
    }
}

impl HasAsyncness for AsyncGenericFn<'_, state::Async> {
    fn asyncness(&self) -> Option<Token![async]> {
        Some(Token![async](Span::call_site()))
    }
}

trait HasIdent {
    fn ident(&self) -> Cow<'_, Ident>;
}

impl HasIdent for AsyncGenericFn<'_, state::Sync> {
    fn ident(&self) -> Cow<'_, Ident> {
        Cow::Borrowed(&self.target.sig().ident)
    }
}

impl HasIdent for AsyncGenericFn<'_, state::Async> {
    fn ident(&self) -> Cow<'_, Ident> {
        (&self.target.sig().ident)
            .r#let(|ident| Cow::Owned(Ident::new(&format!("{ident}_async"), ident.span())))
    }
}

trait HasInputs {
    fn inputs(&self) -> &Punctuated<FnArg, Token![,]>;
}

impl HasInputs for AsyncGenericFn<'_, state::Sync> {
    fn inputs(&self) -> &Punctuated<FnArg, Token![,]> {
        &self.target.sig().inputs
    }
}

impl HasInputs for AsyncGenericFn<'_, state::Async> {
    fn inputs(&self) -> &Punctuated<FnArg, Token![,]> {
        self.state
            .0
            .as_ref()
            .map(|AsyncSignature { inputs, .. }| inputs)
            .unwrap_or_else(|| &self.target.sig().inputs)
    }
}

trait HasGenerics {
    fn generics(&self) -> &Generics;
}

impl HasGenerics for AsyncGenericFn<'_, state::Sync> {
    fn generics(&self) -> &Generics {
        &self.target.sig().generics
    }
}

impl HasGenerics for AsyncGenericFn<'_, state::Async> {
    fn generics(&self) -> &Generics {
        self.state
            .0
            .as_ref()
            .map(|AsyncSignature { generics, .. }| generics)
            .unwrap_or_else(|| &self.target.sig().generics)
    }
}

trait HasOutput {
    fn output(&self) -> &ReturnType;
}

impl<S> HasOutput for AsyncGenericFn<'_, S> {
    fn output(&self) -> &ReturnType {
        &self.target.sig().output
    }
}

trait HasBlock {
    fn block(&self) -> Either<Block, Token![;]>;
}

impl<'f, S> HasBlock for AsyncGenericFn<'f, S>
where
    AsyncGenericFn<'f, S>: CanRewriteBlock,
{
    fn block(&self) -> Either<Block, Token![;]> {
        match self.target.block() {
            Some(block) => Either::A(Self::rewrite_block(block)),
            None => Either::B(Token![;](Span::call_site())),
        }
    }
}

trait CanRewriteBlock {
    fn rewrite_block(block: &Block) -> Block;
}

impl CanRewriteBlock for AsyncGenericFn<'_, state::Sync> {
    fn rewrite_block(_block: &Block) -> Block {
        todo!()
    }
}

impl CanRewriteBlock for AsyncGenericFn<'_, state::Async> {
    fn rewrite_block(_block: &Block) -> Block {
        todo!()
    }
}

struct StmtVisitor;

impl VisitMut for StmtVisitor {
    fn visit_stmt_mut(&mut self, i: &mut Stmt) {
        // visit_mut::visit_stmt_mut()
        todo!()
    }
}

impl<'f, S> ToTokens for AsyncGenericFn<'f, S>
where
    AsyncGenericFn<'f, S>: HasFunctionParts,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let constness = self.constness();
        let asyncness = self.asyncness();
        let unsafety = self.unsafety();
        let ident = self.ident();
        let inputs = self.inputs();
        let generics = self.generics();
        let where_clause = generics.where_clause.as_ref();
        let output = self.output();
        let block = self.block();
        tokens.extend(quote! {
            #constness #asyncness #unsafety fn #ident #generics(#inputs) #output
            #where_clause
            #block
        });
    }
}
