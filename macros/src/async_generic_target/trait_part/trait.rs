use syn::{Attribute, ItemTrait, TraitItem, TraitItemFn};
use crate::core_attr::cfg::CfgAttribute;
use crate::core_attr::cfg_attr::CfgAttrAttribute;
use super::{HasAttributes, TraitPart, TraitPartItem};

impl TraitPart for ItemTrait {
    type Item = TraitItem;

    fn items(&self) -> &[Self::Item] {
        &self.items
    }

    fn try_replace_items<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(Vec<Self::Item>) -> Result<Vec<Self::Item>, E>,
    {
        self.items = f(std::mem::take(&mut self.items))?;
        Ok(())
    }

    fn extend_attrs(&mut self, iter: impl IntoIterator<Item = Attribute>) {
        self.attrs.extend(iter);
    }
}

impl TraitPartItem for TraitItem {
    type ItemFn = TraitItemFn;

    fn as_item_fn(&self) -> Option<&Self::ItemFn> {
        match self {
            TraitItem::Fn(item_fn) => Some(item_fn),
            _ => None,
        }
    }

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

    fn push_cfg(&mut self, cfg: CfgAttribute) {
        self.attrs.push(cfg.into());
    }

    fn push_cfg_attr(&mut self, cfg_attr: CfgAttrAttribute) {
        self.attrs.push(cfg_attr.into());
    }
}
