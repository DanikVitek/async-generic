use proc_macro2::Ident;
use syn::{
    punctuated::Punctuated, Attribute, Generics, ImplItem, ImplItemFn, ItemImpl, Token,
    TypeParamBound,
};

use super::{HasAsyncness, HasAttributes, TraitPart, TraitPartItem};

impl TraitPart for ItemImpl {
    type Item = ImplItem;

    fn update_ident(&mut self, f: impl FnOnce(Ident) -> Ident) {
        if let Some((_, path, _)) = &mut self.trait_ {
            if let Some((mut last_segment, punct)) =
                path.segments.pop().map(|segment| segment.into_tuple())
            {
                last_segment.ident = f(last_segment.ident);
                path.segments.push_value(last_segment);
                if let Some(punct) = punct {
                    path.segments.push_punct(punct);
                }
            }
        }
    }

    fn items(&self) -> &[Self::Item] {
        &self.items
    }

    fn items_mut(&mut self) -> &mut Vec<Self::Item> {
        &mut self.items
    }

    fn set_items(&mut self, items: Vec<Self::Item>) {
        self.items = items;
    }

    fn try_update_items<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(Vec<Self::Item>) -> Result<Vec<Self::Item>, E>,
    {
        self.items = f(std::mem::take(&mut self.items))?;
        Ok(())
    }

    fn extend_attrs(&mut self, iter: impl IntoIterator<Item = Attribute>) {
        self.attrs.extend(iter);
    }

    fn set_colon_token(&mut self, _: Token![:]) {
        // N/A
    }

    fn set_supertraits(&mut self, _: Punctuated<TypeParamBound, Token![+]>) {
        // N/A
    }

    fn set_generics(&mut self, generics: Generics) {
        self.generics = generics;
    }
}

impl TraitPartItem for ImplItem {
    type ItemFn = ImplItemFn;

    fn as_item_fn(&self) -> Option<&Self::ItemFn> {
        match self {
            ImplItem::Fn(item_fn) => Some(item_fn),
            _ => None,
        }
    }

    fn to_item_fn(self) -> Result<Self::ItemFn, Self>
    where
        Self: Sized,
    {
        match self {
            ImplItem::Fn(item_fn) => Ok(item_fn),
            _ => Err(self),
        }
    }
}

impl HasAttributes for ImplItemFn {
    fn attrs(&self) -> &[Attribute] {
        &self.attrs
    }

    fn remove_attr(&mut self, index: usize) -> Attribute {
        self.attrs.remove(index)
    }
}

impl HasAsyncness for ImplItemFn {
    fn is_async(&self) -> bool {
        self.sig.asyncness.is_some()
    }
}
