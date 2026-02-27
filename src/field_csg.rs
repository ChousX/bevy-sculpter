// field_csg.rs
//! Cross-type CSG operations for combining arbitrary Field types.
//!
//! Allows combining fields with different storage types (f32, bool, u8, etc.)
//! by converting through a common signed distance representation via [`IsoConvertible`].

use crate::field::{Field, smooth_max, smooth_min};
use bevy::prelude::*;

// ============================================================================
// Iso conversion
// ============================================================================

/// Maps a storage type to and from signed distance values.
///
/// This is the bridge enabling cross-type CSG. Negative iso = inside,
/// positive iso = outside.
pub trait IsoConvertible: Copy + Default {
    /// Convert a storage value to a signed distance.
    fn to_iso(self) -> f32;
    /// Convert a signed distance back to a storage value.
    fn from_iso(iso: f32) -> Self;
}

impl IsoConvertible for f32 {
    #[inline]
    fn to_iso(self) -> f32 {
        self
    }
    #[inline]
    fn from_iso(iso: f32) -> Self {
        iso
    }
}

impl IsoConvertible for bool {
    #[inline]
    fn to_iso(self) -> f32 {
        if self { -1.0 } else { 1.0 }
    }
    #[inline]
    fn from_iso(iso: f32) -> Self {
        iso < 0.0
    }
}

impl IsoConvertible for u8 {
    #[inline]
    fn to_iso(self) -> f32 {
        if self > 0 { -1.0 } else { 1.0 }
    }
    #[inline]
    fn from_iso(iso: f32) -> Self {
        if iso < 0.0 { 255 } else { 0 }
    }
}

impl IsoConvertible for i8 {
    #[inline]
    fn to_iso(self) -> f32 {
        if self >= 0 {
            self as f32 / 127.0
        } else {
            self as f32 / 128.0
        }
    }

    #[inline]
    fn from_iso(iso: f32) -> Self {
        let clamped = iso.clamp(-1.0, 1.0);
        if clamped >= 0.0 {
            (clamped * 127.0).round() as i8
        } else {
            (clamped * 128.0).round().max(-128.0) as i8
        }
    }
}

// ============================================================================
// CSG operation enum
// ============================================================================

/// CSG combinators operating on iso values.
#[derive(Clone, Copy, Debug)]
pub enum CsgOp {
    /// Combine both volumes: `min(a, b)`
    Union,
    /// Keep only overlap: `max(a, b)`
    Intersect,
    /// Remove b from a: `max(a, -b)`
    Subtract,
    /// Smooth union with blending radius k.
    SmoothUnion(f32),
    /// Smooth intersection with blending radius k.
    SmoothIntersect(f32),
    /// Smooth subtraction with blending radius k.
    SmoothSubtract(f32),
}

impl CsgOp {
    #[inline]
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Union => a.min(b),
            Self::Intersect => a.max(b),
            Self::Subtract => a.max(-b),
            Self::SmoothUnion(k) => smooth_min(a, b, k),
            Self::SmoothIntersect(k) => smooth_max(a, b, k),
            Self::SmoothSubtract(k) => smooth_max(a, -b, k),
        }
    }
}

// ============================================================================
// Extension trait
// ============================================================================

/// CSG operations on any `Field<T>` where `T: IsoConvertible`.
///
/// All methods convert both inputs to iso, apply the CSG operation,
/// then convert to the output type.
///
/// # Examples
///
/// ```ignore
/// use bevy_sculpter::field_csg::FieldCsg;
///
/// // Intersect an SDF with a bool mask, output as SDF
/// let result: SdfVolume = sdf_field.csg_intersect(&bool_mask);
///
/// // Union two bool fields
/// let result: BoolField = field_a.csg_union(&field_b);
///
/// // Custom blend in iso space
/// let result: SdfVolume = field_a.csg_custom(&field_b, |a, b| a * 0.7 + b * 0.3);
/// ```
pub trait FieldCsg<Ta: IsoConvertible>: Field<Ta> {
    /// Combine with another field using a [`CsgOp`].
    fn csg<B, Tb, O, To>(&self, other: &B, op: CsgOp) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        assert_eq!(Self::SIZE, B::SIZE, "Field sizes must match");
        assert_eq!(Self::SIZE, O::SIZE, "Output field size must match");

        let mut result = O::default();
        let a = self.data();
        let b = other.data();
        let out = result.data_mut();

        for i in 0..Self::VOLUME {
            out[i] = To::from_iso(op.apply(a[i].to_iso(), b[i].to_iso()));
        }
        result
    }

    /// CSG union (combine volumes).
    fn csg_union<B, Tb, O, To>(&self, other: &B) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::Union)
    }

    /// CSG intersection (keep overlap).
    fn csg_intersect<B, Tb, O, To>(&self, other: &B) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::Intersect)
    }

    /// CSG subtraction (remove other from self).
    fn csg_subtract<B, Tb, O, To>(&self, other: &B) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::Subtract)
    }

    /// Smooth CSG union.
    fn csg_union_smooth<B, Tb, O, To>(&self, other: &B, k: f32) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::SmoothUnion(k))
    }

    /// Smooth CSG intersection.
    fn csg_intersect_smooth<B, Tb, O, To>(&self, other: &B, k: f32) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::SmoothIntersect(k))
    }

    /// Smooth CSG subtraction.
    fn csg_subtract_smooth<B, Tb, O, To>(&self, other: &B, k: f32) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
    {
        self.csg(other, CsgOp::SmoothSubtract(k))
    }

    /// Combine with a custom closure operating in iso space.
    fn csg_custom<B, Tb, O, To, F>(&self, other: &B, op: F) -> O
    where
        B: Field<Tb>,
        Tb: IsoConvertible,
        O: Field<To>,
        To: IsoConvertible,
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(Self::SIZE, B::SIZE, "Field sizes must match");
        assert_eq!(Self::SIZE, O::SIZE, "Output field size must match");

        let mut result = O::default();
        let a = self.data();
        let b = other.data();
        let out = result.data_mut();

        for i in 0..Self::VOLUME {
            out[i] = To::from_iso(op(a[i].to_iso(), b[i].to_iso()));
        }
        result
    }

    /// Combine with full access to raw storage values.
    ///
    /// Use when you need the original values (e.g. material IDs),
    /// not just their iso representation.
    fn csg_raw<B, Tb, O, To, F>(&self, other: &B, op: F) -> O
    where
        B: Field<Tb>,
        Tb: Copy + Default,
        O: Field<To>,
        To: Copy + Default,
        F: Fn(Ta, Tb) -> To,
    {
        assert_eq!(Self::SIZE, B::SIZE, "Field sizes must match");
        assert_eq!(Self::SIZE, O::SIZE, "Output field size must match");

        let mut result = O::default();
        let a = self.data();
        let b = other.data();
        let out = result.data_mut();

        for i in 0..Self::VOLUME {
            out[i] = op(a[i], b[i]);
        }
        result
    }
}

impl<Ta: IsoConvertible, F: Field<Ta>> FieldCsg<Ta> for F {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct F32Field(Vec<f32>);
    impl Field<f32> for F32Field {
        const SIZE: UVec3 = uvec3(4, 4, 4);
        const DEFAULT: f32 = 1.0;
        fn data(&self) -> &[f32] {
            &self.0
        }
        fn data_mut(&mut self) -> &mut [f32] {
            &mut self.0
        }
    }

    #[derive(Default)]
    struct BoolField(Vec<bool>);
    impl Field<bool> for BoolField {
        const SIZE: UVec3 = uvec3(4, 4, 4);
        const DEFAULT: bool = false;
        fn data(&self) -> &[bool] {
            &self.0
        }
        fn data_mut(&mut self) -> &mut [bool] {
            &mut self.0
        }
    }

    #[test]
    fn test_f32_union() {
        let mut a = F32Field(vec![1.0; 64]);
        let mut b = F32Field(vec![1.0; 64]);
        a.0[F32Field::index(2, 2, 2)] = -1.0;
        b.0[F32Field::index(3, 2, 2)] = -1.0;

        let result: F32Field = a.csg_union(&b);
        assert!(result.get(2, 2, 2) < 0.0);
        assert!(result.get(3, 2, 2) < 0.0);
        assert!(result.get(0, 0, 0) > 0.0);
    }

    #[test]
    fn test_bool_f32_intersect() {
        let mut mask = BoolField(vec![false; 64]);
        let sdf = F32Field(vec![-1.0; 64]);
        for i in 0..32 {
            mask.0[i] = true;
        }

        let result: F32Field = mask.csg_intersect(&sdf);
        assert!(result.data()[0] < 0.0);
        assert!(result.data()[32] > 0.0);
    }

    #[test]
    fn test_cross_type_subtract() {
        let base = F32Field(vec![-1.0; 64]);
        let mut cutter = BoolField(vec![false; 64]);
        cutter.0[BoolField::index(2, 2, 2)] = true;

        let result: F32Field = base.csg_subtract(&cutter);
        assert!(result.get(2, 2, 2) > 0.0);
        assert!(result.get(0, 0, 0) < 0.0);
    }

    #[test]
    fn test_output_as_bool() {
        let mut a = F32Field(vec![1.0; 64]);
        let mut b = F32Field(vec![1.0; 64]);
        a.0[F32Field::index(1, 1, 1)] = -1.0;
        b.0[F32Field::index(1, 1, 1)] = -1.0;
        b.0[F32Field::index(2, 2, 2)] = -1.0;

        let result: BoolField = a.csg_union(&b);
        assert!(result.get(1, 1, 1));
        assert!(result.get(2, 2, 2));
        assert!(!result.get(0, 0, 0));
    }

    #[test]
    fn test_custom_blend() {
        let a = F32Field(vec![-1.0; 64]);
        let b = F32Field(vec![0.5; 64]);

        let result: F32Field = a.csg_custom(&b, |a, b| a * 0.5 + b * 0.5);
        assert!((result.data()[0] - (-0.25)).abs() < 0.001);
    }

    #[test]
    fn test_raw_combiner() {
        let a = F32Field(vec![-1.0; 64]);
        let b = F32Field(vec![0.5; 64]);

        let result: F32Field = a.csg_raw(&b, |va, vb| (va + vb) * 0.5);
        assert!((result.data()[0] - (-0.25)).abs() < 0.001);
    }
}
