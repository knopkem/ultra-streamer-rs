/// A packetized fragment of an encoded video frame, sized to fit in a QUIC datagram.
#[derive(Debug, Clone)]
pub struct FramePacket {
    /// Monotonically increasing frame counter.
    pub frame_id: u32,
    /// Fragment index within this frame (0..fragment_count).
    pub fragment_idx: u16,
    /// Total number of fragments for this frame.
    pub fragment_count: u16,
    /// Capture timestamp in microseconds (relative to session start).
    pub timestamp_us: u64,
    /// Whether this is a keyframe (IDR).
    pub is_keyframe: bool,
    /// Whether this frame is lossless (diagnostic refinement).
    pub is_lossless: bool,
    /// Encoded NALU fragment payload.
    pub payload: Vec<u8>,
}

/// Header size in bytes for the binary wire format.
pub const FRAME_PACKET_HEADER_SIZE: usize = 18;

/// Maximum payload size per datagram (conservative for LAN, can increase with jumbo frames).
pub const MAX_DATAGRAM_PAYLOAD: usize = 1200 - FRAME_PACKET_HEADER_SIZE;

impl FramePacket {
    /// Serialize to binary wire format for QUIC datagram.
    ///
    /// Layout (18 bytes header + payload):
    /// ```text
    /// [0..4]   frame_id: u32 LE
    /// [4..6]   fragment_idx: u16 LE
    /// [6..8]   fragment_count: u16 LE
    /// [8..16]  timestamp_us: u64 LE
    /// [16]     flags: u8 (bit 0 = keyframe, bit 1 = lossless)
    /// [17]     reserved: u8
    /// [18..]   payload
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FRAME_PACKET_HEADER_SIZE + self.payload.len());
        buf.extend_from_slice(&self.frame_id.to_le_bytes());
        buf.extend_from_slice(&self.fragment_idx.to_le_bytes());
        buf.extend_from_slice(&self.fragment_count.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_us.to_le_bytes());
        let flags = (self.is_keyframe as u8) | ((self.is_lossless as u8) << 1);
        buf.push(flags);
        buf.push(0); // reserved
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Deserialize from binary wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FramePacketError> {
        if data.len() < FRAME_PACKET_HEADER_SIZE {
            return Err(FramePacketError::TooShort {
                len: data.len(),
                expected: FRAME_PACKET_HEADER_SIZE,
            });
        }

        let frame_id = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let fragment_idx = u16::from_le_bytes(data[4..6].try_into().unwrap());
        let fragment_count = u16::from_le_bytes(data[6..8].try_into().unwrap());
        let timestamp_us = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let flags = data[16];

        Ok(Self {
            frame_id,
            fragment_idx,
            fragment_count,
            timestamp_us,
            is_keyframe: flags & 0x01 != 0,
            is_lossless: flags & 0x02 != 0,
            payload: data[FRAME_PACKET_HEADER_SIZE..].to_vec(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FramePacketError {
    #[error("packet too short: {len} bytes, expected at least {expected}")]
    TooShort { len: usize, expected: usize },
}

/// Split an encoded frame (NALUs) into datagram-sized `FramePacket`s.
pub fn packetize_frame(
    frame_id: u32,
    timestamp_us: u64,
    is_keyframe: bool,
    is_lossless: bool,
    nalu_data: &[u8],
) -> Vec<FramePacket> {
    let chunks: Vec<&[u8]> = nalu_data.chunks(MAX_DATAGRAM_PAYLOAD).collect();
    let fragment_count = chunks.len() as u16;

    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| FramePacket {
            frame_id,
            fragment_idx: i as u16,
            fragment_count,
            timestamp_us,
            is_keyframe,
            is_lossless,
            payload: chunk.to_vec(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_frame_packet() {
        let packet = FramePacket {
            frame_id: 42,
            fragment_idx: 1,
            fragment_count: 3,
            timestamp_us: 123456789,
            is_keyframe: true,
            is_lossless: false,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        let bytes = packet.to_bytes();
        let decoded = FramePacket::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.frame_id, 42);
        assert_eq!(decoded.fragment_idx, 1);
        assert_eq!(decoded.fragment_count, 3);
        assert_eq!(decoded.timestamp_us, 123456789);
        assert!(decoded.is_keyframe);
        assert!(!decoded.is_lossless);
        assert_eq!(decoded.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn packetize_splits_correctly() {
        let data = vec![0u8; MAX_DATAGRAM_PAYLOAD * 2 + 100];
        let packets = packetize_frame(1, 0, true, false, &data);
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].fragment_count, 3);
        assert_eq!(packets[0].fragment_idx, 0);
        assert_eq!(packets[2].fragment_idx, 2);
        assert_eq!(packets[2].payload.len(), 100);
    }
}
