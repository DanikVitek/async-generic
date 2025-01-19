pub trait LetExt: Sized {
    #[inline]
    fn r#let<F, T>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T,
    {
        f(self)
    }
}

impl<A> LetExt for A {}

pub trait InspectExt<T>: Sized {
    // TODO: remove once MSRV >= 1.76
    fn inspect<F: FnOnce(&T)>(self, f: F) -> Self;
}

impl<T, E> InspectExt<T> for Result<T, E> {
    #[inline]
    fn inspect<F: FnOnce(&T)>(self, f: F) -> Self {
        if let Ok(ref t) = self {
            f(t);
        }

        self
    }
}
