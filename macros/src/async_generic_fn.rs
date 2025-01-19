use std::{borrow::Cow, marker::PhantomData};

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
    Block, Expr, ExprBlock, FnArg, Generics, ReturnType, Token, Visibility,
};

use crate::{
    util::{Either, LetExt},
    AsyncSignature, TargetFn,
};

pub struct AsyncGenericFn<'f, S> {
    target: &'f TargetFn,
    state: S,
}

impl<'f> AsyncGenericFn<'f, state::Sync> {
    pub(crate) const fn new(target: &'f TargetFn) -> Self {
        Self {
            target,
            state: state::Sync,
        }
    }
}

impl<'f> AsyncGenericFn<'f, state::Async> {
    pub(crate) const fn new(target: &'f TargetFn, sig: Option<AsyncSignature>) -> Self {
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
    HasVisibility
    + HasConstness
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
    T: HasVisibility
        + HasConstness
        + HasUnsafety
        + HasAsyncness
        + HasIdent
        + HasInputs
        + HasGenerics
        + HasOutput
        + HasBlock
{
}

trait HasVisibility {
    fn visibility(&self) -> Option<&Visibility>;
}

impl<S> HasVisibility for AsyncGenericFn<'_, S> {
    fn visibility(&self) -> Option<&Visibility> {
        self.target.visibility()
    }
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

impl<S> HasBlock for AsyncGenericFn<'_, S>
where
    IfAsyncRewriter<S>: CanRewriteBlock,
{
    fn block(&self) -> Either<Block, Token![;]> {
        match self.target.block() {
            Some(block) => {
                let mut block = block.clone();
                IfAsyncRewriter::<S>(PhantomData).visit_block_mut(&mut block);
                Either::A(block)
            }
            None => Either::B(Token![;](Span::call_site())),
        }
    }
}

trait CanRewriteBlock {
    fn rewrite_block(
        node: &mut Expr,
        predicate: AsyncPredicate,
        then_branch: Block,
        else_branch: Option<Expr>,
    );
}

enum AsyncPredicate {
    Sync,
    Async,
}

struct IfAsyncRewriter<S>(PhantomData<S>);

trait CanCompareToPredicate {
    fn cmp(predicate: AsyncPredicate) -> bool;
}

impl CanCompareToPredicate for state::Sync {
    fn cmp(predicate: AsyncPredicate) -> bool {
        matches!(predicate, AsyncPredicate::Sync)
    }
}

impl CanCompareToPredicate for state::Async {
    fn cmp(predicate: AsyncPredicate) -> bool {
        matches!(predicate, AsyncPredicate::Async)
    }
}

impl<S> CanRewriteBlock for IfAsyncRewriter<S>
where
    S: CanCompareToPredicate,
{
    fn rewrite_block(
        node: &mut Expr,
        predicate: AsyncPredicate,
        then_branch: Block,
        else_branch: Option<Expr>,
    ) {
        *node = if S::cmp(predicate) {
            Expr::Block(ExprBlock {
                attrs: vec![],
                label: None,
                block: then_branch,
            })
        } else if let Some(else_expr) = else_branch {
            else_expr
        } else {
            parse_quote! {{}}
        }
    }
}

impl<S> VisitMut for IfAsyncRewriter<S>
where
    IfAsyncRewriter<S>: CanRewriteBlock,
{
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        visit_mut::visit_expr_mut(self, node);

        let Expr::If(expr_if) = node else {
            return;
        };
        let Expr::Path(cond) = expr_if.cond.as_ref() else {
            return;
        };
        if !cond.attrs.is_empty()
            || !cond.qself.is_none()
            || cond.path.leading_colon.is_some()
            || cond.path.segments.len() != 1
        {
            return;
        }
        let segment = cond.path.segments.first().unwrap();
        if !segment.arguments.is_none() {
            return;
        }
        let predicate = match segment.ident.to_string().as_str() {
            "_sync" => AsyncPredicate::Sync,
            "_async" => AsyncPredicate::Async,
            _ => return,
        };
        let then_branch = expr_if.then_branch.clone();
        let else_branch = expr_if.else_branch.as_ref().map(|eb| *eb.1.clone());

        Self::rewrite_block(node, predicate, then_branch, else_branch);
    }
}

impl<'f, S> ToTokens for AsyncGenericFn<'f, S>
where
    AsyncGenericFn<'f, S>: HasFunctionParts,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let visibility = self.visibility();
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
            #visibility #constness #asyncness #unsafety fn #ident #generics(#inputs) #output
            #where_clause
            #block
        });
    }
}
