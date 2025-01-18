use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;

pub trait LetExt {
    #[inline]
    fn r#let<F, T>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T,
        Self: Sized,
    {
        f(self)
    }

    #[inline]
    fn let_ref<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn let_mut<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }
}

impl<A> LetExt for A {}

pub enum Either<A, B> {
    A(A),
    B(B),
}

impl<A: ToTokens, B: ToTokens> ToTokens for Either<A, B> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::A(a) => a.to_tokens(tokens),
            Self::B(b) => b.to_tokens(tokens),
        }
    }
}
