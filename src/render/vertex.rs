#[derive(Copy, Clone, Debug)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Indices {
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        match self {
            Indices::U16(vec) => IndicesIter::U16(vec.iter()),
            Indices::U32(vec) => IndicesIter::U32(vec.iter()),
        }
    }

    pub fn get_total_size(&self) -> usize {
        match self {
            Indices::U16(_) => std::mem::size_of::<u16>() * self.len(),
            Indices::U32(_) => std::mem::size_of::<u32>() * self.len(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Indices::U16(ref values) => values.len(),
            Indices::U32(ref values) => values.len(),
        }
    }
}

enum IndicesIter<'a> {
    U16(std::slice::Iter<'a, u16>),
    U32(std::slice::Iter<'a, u32>),
}

impl Iterator for IndicesIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndicesIter::U16(iter) => iter.next().map(|val| *val as usize),
            IndicesIter::U32(iter) => iter.next().map(|val| *val as usize),
        }
    }
}
