use proc_macro2::Ident;
use syn::{
    punctuated::Punctuated, Attribute, Generics, ItemTrait, Token, TraitItem, TraitItemFn,
    TypeParamBound,
};

use super::{HasAsyncness, HasAttributes, TraitPart, TraitPartItem};

impl TraitPart for ItemTrait {
    type Item = TraitItem;

    fn update_ident(&mut self, f: impl FnOnce(Ident) -> Ident) {
        self.ident = f(self.ident.clone());
    }

    fn items_mut(&mut self) -> &mut Vec<Self::Item> {
        &mut self.items
    }

    fn set_items(&mut self, items: Vec<Self::Item>) {
        self.items = items;
    }

    fn extend_attrs(&mut self, iter: impl IntoIterator<Item = Attribute>) {
        self.attrs.extend(iter);
    }

    fn set_colon_token(&mut self, colon_token: Token![:]) {
        self.colon_token = Some(colon_token);
    }

    fn set_supertraits(&mut self, supertraits: Punctuated<TypeParamBound, Token![+]>) {
        self.supertraits = supertraits;
    }

    fn set_generics(&mut self, generics: Generics) {
        self.generics = generics;
    }
}

impl TraitPartItem for TraitItem {
    type ItemFn = TraitItemFn;

    fn to_item_fn(self) -> Result<Self::ItemFn, Self>
    where
        Self: Sized,
    {
        match self {
            TraitItem::Fn(item_fn) => Ok(item_fn),
            _ => Err(self),
        }
    }
}

impl HasAttributes for TraitItemFn {
    fn attrs(&self) -> &[Attribute] {
        &self.attrs
    }

    fn remove_attr(&mut self, i: usize) -> Attribute {
        self.attrs.remove(i)
    }
}

impl HasAsyncness for TraitItemFn {
    fn is_async(&self) -> bool {
        self.sig.asyncness.is_some()
    }
}
