use super::{AsyncTrait, SyncTrait};

pub struct Sync<const SPLIT_FNS: bool>(pub(super) Option<SyncTrait>);

pub struct Async(pub(super) AsyncTrait);
