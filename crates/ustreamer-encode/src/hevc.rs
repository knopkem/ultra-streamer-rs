use crate::DecoderConfig;

pub(crate) const HEVC_NAL_TYPE_VPS: u8 = 32;
pub(crate) const HEVC_NAL_TYPE_SPS: u8 = 33;
pub(crate) const HEVC_NAL_TYPE_PPS: u8 = 34;
const HEVC_ACCESS_UNIT_LENGTH_BYTES: usize = 4;
pub(crate) const HEVC_HVCC_LENGTH_SIZE_MINUS_ONE: u8 = (HEVC_ACCESS_UNIT_LENGTH_BYTES - 1) as u8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HevcParameterSets {
    pub(crate) vps: Vec<u8>,
    pub(crate) sps: Vec<u8>,
    pub(crate) pps: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HevcSpsMetadata {
    general_profile_space: u8,
    general_tier_flag: bool,
    general_profile_idc: u8,
    general_profile_compatibility_flags: u32,
    general_constraint_indicator_flags: u64,
    general_level_idc: u8,
    chroma_format_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    num_temporal_layers: u8,
    temporal_id_nested: bool,
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_offset: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_offset: 0,
        }
    }

    fn read_bit(&mut self) -> Result<u8, String> {
        self.read_bits(1).map(|value| value as u8)
    }

    fn read_bits(&mut self, bit_count: u8) -> Result<u64, String> {
        let mut value = 0u64;
        for _ in 0..bit_count {
            if self.bit_offset >= self.data.len().saturating_mul(8) {
                return Err("unexpected end of bitstream".into());
            }
            let byte = self.data[self.bit_offset / 8];
            let shift = 7 - (self.bit_offset % 8);
            value = (value << 1) | u64::from((byte >> shift) & 1);
            self.bit_offset += 1;
        }
        Ok(value)
    }

    fn read_uvlc(&mut self) -> Result<u32, String> {
        let mut leading_zero_bits = 0u32;
        while self.read_bit()? == 0 {
            leading_zero_bits = leading_zero_bits.saturating_add(1);
            if leading_zero_bits > 31 {
                return Err("unsigned variable-length code exceeds supported range".into());
            }
        }
        let suffix = if leading_zero_bits == 0 {
            0
        } else {
            self.read_bits(leading_zero_bits as u8)? as u32
        };
        Ok((1u32 << leading_zero_bits) - 1 + suffix)
    }

    fn read_ue(&mut self) -> Result<u32, String> {
        self.read_uvlc()
    }
}

pub(crate) fn normalize_hevc_access_unit(data: &[u8]) -> Result<Vec<u8>, String> {
    let nal_units = extract_annex_b_nalus(data);
    if nal_units.is_empty() {
        return Ok(data.to_vec());
    }

    let mut output = Vec::with_capacity(data.len());
    for nal_unit in nal_units {
        let length = u32::try_from(nal_unit.len())
            .map_err(|_| "HEVC NAL unit exceeded 32-bit length field".to_string())?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(nal_unit);
    }
    Ok(output)
}

pub(crate) fn decoder_config_from_hevc_access_unit(
    access_unit: &[u8],
    width: u32,
    height: u32,
) -> Result<DecoderConfig, String> {
    Ok(DecoderConfig {
        codec: build_hevc_codec_string_from_sequence_payload(access_unit)?,
        description: Some(build_hevc_hvcc_description(access_unit)?),
        coded_width: width,
        coded_height: height,
    })
}

pub(crate) fn build_hevc_hvcc_description(sequence_payload: &[u8]) -> Result<Vec<u8>, String> {
    let parameter_sets = extract_hevc_parameter_sets(sequence_payload)?;
    let metadata = parse_hevc_sps_metadata(&parameter_sets.sps)?;

    let mut description = Vec::with_capacity(
        23 + parameter_sets.vps.len() + parameter_sets.sps.len() + parameter_sets.pps.len() + 18,
    );
    description.push(1);
    description.push(
        ((metadata.general_profile_space & 0x03) << 6)
            | (u8::from(metadata.general_tier_flag) << 5)
            | (metadata.general_profile_idc & 0x1f),
    );
    description.extend_from_slice(&metadata.general_profile_compatibility_flags.to_be_bytes());
    description.extend_from_slice(&metadata.general_constraint_indicator_flags.to_be_bytes()[2..]);
    description.push(metadata.general_level_idc);
    description.extend_from_slice(&0xF000u16.to_be_bytes());
    description.push(0xFC);
    description.push(0xFC | (metadata.chroma_format_idc & 0x03));
    description.push(0xF8 | (metadata.bit_depth_luma_minus8 & 0x07));
    description.push(0xF8 | (metadata.bit_depth_chroma_minus8 & 0x07));
    description.extend_from_slice(&0u16.to_be_bytes());
    description.push(
        ((metadata.num_temporal_layers.max(1).min(7) & 0x07) << 3)
            | (u8::from(metadata.temporal_id_nested) << 2)
            | (HEVC_HVCC_LENGTH_SIZE_MINUS_ONE & 0x03),
    );
    description.push(3);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_VPS, &parameter_sets.vps);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_SPS, &parameter_sets.sps);
    append_hvcc_array(&mut description, HEVC_NAL_TYPE_PPS, &parameter_sets.pps);
    Ok(description)
}

pub(crate) fn build_hevc_codec_string_from_sequence_payload(
    sequence_payload: &[u8],
) -> Result<String, String> {
    let parameter_sets = extract_hevc_parameter_sets(sequence_payload)?;
    let metadata = parse_hevc_sps_metadata(&parameter_sets.sps)?;
    Ok(build_hevc_codec_string(metadata))
}

fn build_hevc_codec_string(metadata: HevcSpsMetadata) -> String {
    let mut codec = String::from("hvc1.");
    match metadata.general_profile_space {
        1 => codec.push('A'),
        2 => codec.push('B'),
        3 => codec.push('C'),
        _ => {}
    }
    codec.push_str(&metadata.general_profile_idc.to_string());
    codec.push('.');
    codec.push_str(
        &metadata
            .general_profile_compatibility_flags
            .reverse_bits()
            .to_string(),
    );
    codec.push('.');
    codec.push(if metadata.general_tier_flag { 'H' } else { 'L' });
    codec.push_str(&metadata.general_level_idc.to_string());

    let mut constraint_bytes =
        metadata.general_constraint_indicator_flags.to_be_bytes()[2..].to_vec();
    while constraint_bytes.last() == Some(&0) {
        constraint_bytes.pop();
    }
    if !constraint_bytes.is_empty() {
        codec.push('.');
        for byte in constraint_bytes {
            use std::fmt::Write as _;
            let _ = write!(codec, "{byte:02X}");
        }
    }

    codec
}

fn append_hvcc_array(description: &mut Vec<u8>, nal_type: u8, nal_unit: &[u8]) {
    description.push(0x80 | (nal_type & 0x3f));
    description.extend_from_slice(&1u16.to_be_bytes());
    description.extend_from_slice(&(nal_unit.len().min(u16::MAX as usize) as u16).to_be_bytes());
    description.extend_from_slice(nal_unit);
}

pub(crate) fn extract_hevc_parameter_sets(
    sequence_payload: &[u8],
) -> Result<HevcParameterSets, String> {
    let mut vps = None;
    let mut sps = None;
    let mut pps = None;

    for nal_unit in extract_annex_b_nalus(sequence_payload) {
        let Some(nal_type) = hevc_nal_type(nal_unit) else {
            continue;
        };
        match nal_type {
            HEVC_NAL_TYPE_VPS if vps.is_none() => vps = Some(nal_unit.to_vec()),
            HEVC_NAL_TYPE_SPS if sps.is_none() => sps = Some(nal_unit.to_vec()),
            HEVC_NAL_TYPE_PPS if pps.is_none() => pps = Some(nal_unit.to_vec()),
            _ => {}
        }
    }

    Ok(HevcParameterSets {
        vps: vps
            .ok_or_else(|| "HEVC sequence payload did not contain a VPS NAL unit".to_string())?,
        sps: sps
            .ok_or_else(|| "HEVC sequence payload did not contain a SPS NAL unit".to_string())?,
        pps: pps
            .ok_or_else(|| "HEVC sequence payload did not contain a PPS NAL unit".to_string())?,
    })
}

fn hevc_nal_type(nal_unit: &[u8]) -> Option<u8> {
    nal_unit.first().map(|byte| (byte >> 1) & 0x3f)
}

pub(crate) fn extract_annex_b_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut nal_units = Vec::new();
    let mut search_from = 0usize;

    while let Some((start_code_offset, start_code_len)) = find_annex_b_start_code(data, search_from)
    {
        let nal_start = start_code_offset + start_code_len;
        let next_start = find_annex_b_start_code(data, nal_start)
            .map(|(offset, _)| offset)
            .unwrap_or(data.len());
        let mut nal_end = next_start;
        while nal_end > nal_start && data[nal_end - 1] == 0 {
            nal_end -= 1;
        }
        if nal_end > nal_start {
            nal_units.push(&data[nal_start..nal_end]);
        }
        search_from = next_start;
    }

    nal_units
}

fn find_annex_b_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut index = from;
    while index + 3 <= data.len() {
        if data[index] == 0 && data[index + 1] == 0 {
            if data.get(index + 2) == Some(&1) {
                return Some((index, 3));
            }
            if data.get(index + 2) == Some(&0) && data.get(index + 3) == Some(&1) {
                return Some((index, 4));
            }
        }
        index += 1;
    }
    None
}

fn parse_hevc_sps_metadata(sps_nal_unit: &[u8]) -> Result<HevcSpsMetadata, String> {
    if sps_nal_unit.len() < 3 {
        return Err("HEVC SPS NAL unit was too short".into());
    }
    let rbsp = remove_emulation_prevention_bytes(&sps_nal_unit[2..]);
    let mut bits = BitReader::new(&rbsp);

    let _sps_video_parameter_set_id = bits.read_bits(4)?;
    let sps_max_sub_layers_minus1 = bits.read_bits(3)? as u8;
    let temporal_id_nested = bits.read_bit()? != 0;
    let general_profile_space = bits.read_bits(2)? as u8;
    let general_tier_flag = bits.read_bit()? != 0;
    let general_profile_idc = bits.read_bits(5)? as u8;
    let general_profile_compatibility_flags = bits.read_bits(32)? as u32;
    let general_constraint_indicator_flags = bits.read_bits(48)?;
    let general_level_idc = bits.read_bits(8)? as u8;

    let mut sub_layer_profile_present_flags =
        Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    let mut sub_layer_level_present_flags = Vec::with_capacity(sps_max_sub_layers_minus1 as usize);
    for _ in 0..sps_max_sub_layers_minus1 {
        sub_layer_profile_present_flags.push(bits.read_bit()? != 0);
        sub_layer_level_present_flags.push(bits.read_bit()? != 0);
    }
    if sps_max_sub_layers_minus1 > 0 {
        for _ in sps_max_sub_layers_minus1..8 {
            let _reserved_zero_2bits = bits.read_bits(2)?;
        }
    }
    for (profile_present, level_present) in sub_layer_profile_present_flags
        .into_iter()
        .zip(sub_layer_level_present_flags.into_iter())
    {
        if profile_present {
            let _sub_layer_profile_space = bits.read_bits(2)?;
            let _sub_layer_tier_flag = bits.read_bit()?;
            let _sub_layer_profile_idc = bits.read_bits(5)?;
            let _sub_layer_profile_compatibility_flags = bits.read_bits(32)?;
            let _sub_layer_constraint_indicator_flags = bits.read_bits(48)?;
        }
        if level_present {
            let _sub_layer_level_idc = bits.read_bits(8)?;
        }
    }

    let _sps_seq_parameter_set_id = bits.read_ue()?;
    let chroma_format_idc = bits.read_ue()?.min(3) as u8;
    if chroma_format_idc == 3 {
        let _separate_colour_plane_flag = bits.read_bit()?;
    }
    let _pic_width_in_luma_samples = bits.read_ue()?;
    let _pic_height_in_luma_samples = bits.read_ue()?;
    let conformance_window_flag = bits.read_bit()? != 0;
    if conformance_window_flag {
        let _left = bits.read_ue()?;
        let _right = bits.read_ue()?;
        let _top = bits.read_ue()?;
        let _bottom = bits.read_ue()?;
    }
    let bit_depth_luma_minus8 = bits.read_ue()?.min(7) as u8;
    let bit_depth_chroma_minus8 = bits.read_ue()?.min(7) as u8;

    Ok(HevcSpsMetadata {
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags,
        general_constraint_indicator_flags,
        general_level_idc,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        num_temporal_layers: sps_max_sub_layers_minus1.saturating_add(1),
        temporal_id_nested,
    })
}

fn remove_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut consecutive_zeros = 0u8;

    for &byte in data {
        if consecutive_zeros >= 2 && byte == 0x03 {
            consecutive_zeros = 0;
            continue;
        }
        rbsp.push(byte);
        if byte == 0 {
            consecutive_zeros = consecutive_zeros.saturating_add(1);
        } else {
            consecutive_zeros = 0;
        }
    }

    rbsp
}
