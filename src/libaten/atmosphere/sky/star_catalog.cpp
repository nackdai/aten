#include "atmosphere/sky/star_catalog.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string_view>

#include "atmosphere/sky/sky_constants.h"
#include "atmosphere/sky/sky_render.h"
#include "math/math.h"

namespace aten::sky {
    namespace {
        std::string Trim(std::string_view value)
        {
            auto begin = value.begin();
            auto end = value.end();

            while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) {
                ++begin;
            }
            while (begin != end && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
                --end;
            }

            return std::string(begin, end);
        }

        std::string Field(std::string_view line, const size_t begin_1based, const size_t end_1based)
        {
            // Bright Star Catalog ReadMe byte ranges are written as 1-based,
            // inclusive columns. Keep the helper in that convention so the code
            // below can mirror the catalog documentation directly.
            if (begin_1based == 0 || end_1based < begin_1based || line.size() < begin_1based) {
                return {};
            }

            const auto begin = begin_1based - 1;
            const auto count = std::min(end_1based, line.size()) - begin;
            return Trim(line.substr(begin, count));
        }

        bool ParseInt(std::string_view text, int32_t& value)
        {
            const auto field = Trim(text);
            if (field.empty()) {
                return false;
            }

            try {
                value = std::stoi(field);
                return true;
            }
            catch (...) {
                return false;
            }
        }

        bool ParseFloat(std::string_view text, float& value)
        {
            const auto field = Trim(text);
            if (field.empty()) {
                return false;
            }

            try {
                value = std::stof(field);
                return true;
            }
            catch (...) {
                return false;
            }
        }

        float ComputeStarIrradianceFromMagnitude(const float visual_magnitude)
        {
            // Paper Eq. (7): Es = 10^(0.4 * (-mv - 9)) W/m2.
            return aten::pow(10.0F, 0.4F * (-visual_magnitude - 9.0F));
        }

        float ComputeEffectiveTemperatureFromBv(const float bv_color)
        {
            // Ballesteros' B-V to effective temperature approximation.
            // The night-sky paper uses Teff = 7000K / (B-V + 0.56), but this
            // broader empirical approximation is less aggressive for very blue
            // and very red stars while matching the Sun closely.
            const auto denom0 = aten::max(0.92F * bv_color + 1.7F, 1.0e-4F);
            const auto denom1 = aten::max(0.92F * bv_color + 0.62F, 1.0e-4F);
            return 4600.0F * (1.0F / denom0 + 1.0F / denom1);
        }

        float Planck(const float wavelength_nm, const float temperature)
        {
            // Spectral radiance of a blackbody. The absolute scale is normalized
            // away in ComputeBlackbodyColor; here we only need the spectral shape.
            //
            // This is not a night-sky paper equation. After the paper estimates
            // Teff from B-V, we use Planck's law to turn that temperature into a
            // visible spectral shape:
            //   B(lambda, T) = (2 h c^2 / lambda^5) / (exp(h c / (lambda k T)) - 1).
            // The exponent below is the h c / (lambda k T) term.
            constexpr auto h = 6.62607015e-34;
            constexpr auto c = 299792458.0;
            constexpr auto k = 1.380649e-23;

            const auto wavelength_m = static_cast<double>(wavelength_nm) * 1.0e-9;
            const auto exponent = h * c / (wavelength_m * k * static_cast<double>(temperature));
            const auto numerator = 2.0 * h * c * c;
            const auto denominator = std::pow(wavelength_m, 5.0) * (std::exp(exponent) - 1.0);

            return static_cast<float>(numerator / denominator);
        }

        aten::vec3 ComputeBlackbodyColor(const float temperature)
        {
            std::vector<float> wavelengths;
            std::vector<float> spectrum;

            wavelengths.reserve((LambdaMax - LambdaMin) / 5 + 1);
            spectrum.reserve((LambdaMax - LambdaMin) / 5 + 1);

            float max_value = 0.0F;
            for (int32_t lambda = LambdaMin; lambda <= LambdaMax; lambda += 5) {
                const auto value = Planck(static_cast<float>(lambda), temperature);
                wavelengths.push_back(static_cast<float>(lambda));
                spectrum.push_back(value);
                max_value = aten::max(max_value, value);
            }

            if (max_value > 0.0F) {
                // Normalize before conversion so star brightness stays controlled
                // by Vmag-derived irradiance, not by blackbody absolute radiance.
                for (auto& value : spectrum) {
                    value /= max_value;
                }
            }

            auto color = ConvertSpectrumToLinearSrgb(wavelengths, spectrum);
            const auto max_component = aten::max(color.x, aten::max(color.y, color.z));
            if (max_component > 0.0F) {
                color = color / max_component;
            }

            color.x = aten::max(color.x, 0.0F);
            color.y = aten::max(color.y, 0.0F);
            color.z = aten::max(color.z, 0.0F);

            return color;
        }

        float EstimateBvFromSpectralType(std::string_view spectral_type)
        {
            // The common BSC5 binary record only stores a two-character spectral
            // shorthand and no B-V color index. Use a coarse main-class estimate
            // so the binary path can still produce plausible star colors.
            //
            // The values below are representative Johnson B-V color indices for
            // main-sequence calibration stars by spectral class. They follow the
            // usual hot-to-cool order:
            //   O5V ~= -0.33, B0V ~= -0.30, A0V ~= -0.02,
            //   F0V ~=  0.30, G0V ~=  0.58, K0V ~=  0.81, M0V ~= 1.40.
            // B is set to an intermediate blue-star value rather than B0V, since
            // the two-character binary shorthand does not give enough detail to
            // distinguish B0 from later B subtypes.
            //
            // This is intentionally only a fallback. The fixed-width ASCII BSC
            // path should be preferred when accurate B-V values are needed,
            // because it provides the per-star B-V field directly.
            if (spectral_type.empty()) {
                return 0.0F;
            }

            switch (std::toupper(static_cast<unsigned char>(spectral_type.front()))) {
            case 'O':
                return -0.33F;
            case 'B':
                return -0.16F;
            case 'A':
                return 0.0F;
            case 'F':
                return 0.30F;
            case 'G':
                return 0.58F;
            case 'K':
                return 0.81F;
            case 'M':
                return 1.40F;
            default:
                return 0.0F;
            }
        }

        /**
         * @brief Convert J2000 right ascension / declination fields to a unit direction.
         *
         * The fixed-width ASCII Bright Star Catalog stores right ascension as
         * hour/minute/second and declination as sign/degree/arcminute/arcsecond.
         * This helper converts those catalog fields into the renderer's J2000
         * equatorial Cartesian direction:
         *   x = cos(dec) cos(ra), y = sin(dec), z = cos(dec) sin(ra).
         *
         * @param ra_h Right ascension hour component. 24 hours cover 360 degrees.
         * @param ra_m Right ascension minute component.
         * @param ra_s Right ascension second component.
         * @param dec_sign Declination sign. Use +1 for north and -1 for south.
         * @param dec_deg Absolute declination degree component.
         * @param dec_min Declination arcminute component.
         * @param dec_sec Declination arcsecond component.
         * @return Normalized J2000 equatorial direction vector.
         */
        aten::vec3 ComputeDirectionFromRaDec(
            const int32_t ra_h,
            const int32_t ra_m,
            const float ra_s,
            const int32_t dec_sign,
            const int32_t dec_deg,
            const int32_t dec_min,
            const int32_t dec_sec)
        {
            // ASCII BSC stores RA as hour/minute/second. One hour corresponds to
            // 15 degrees because 24 hours cover the full 360-degree sky.
            const auto ra_deg = 15.0F * (static_cast<float>(ra_h) + static_cast<float>(ra_m) / 60.0F + ra_s / 3600.0F);

            // Dec is stored as sign + degree/arcmin/arcsec.
            const auto dec_abs_deg = static_cast<float>(dec_deg) + static_cast<float>(dec_min) / 60.0F + static_cast<float>(dec_sec) / 3600.0F;
            const auto dec_deg_signed = static_cast<float>(dec_sign) * dec_abs_deg;

            const auto ra = aten::Deg2Rad(ra_deg);
            const auto dec = aten::Deg2Rad(dec_deg_signed);
            const auto cos_dec = aten::cos(dec);

            return aten::normalize(aten::vec3(
                cos_dec * aten::cos(ra),
                aten::sin(dec),
                cos_dec * aten::sin(ra)));
        }

        /**
         * @brief Convert J2000 right ascension / declination radians to a unit direction.
         *
         * The common BSC5 binary record stores right ascension and declination
         * directly in radians, so this helper skips the hour/degree field parsing
         * used by the ASCII path and applies the same equatorial spherical-to-
         * Cartesian conversion:
         *   x = cos(dec) cos(ra), y = sin(dec), z = cos(dec) sin(ra).
         *
         * @param ra Right ascension in radians, J2000.
         * @param dec Declination in radians, J2000.
         * @return Normalized J2000 equatorial direction vector.
         */
        aten::vec3 ComputeDirectionFromRaDecRadians(const float ra, const float dec)
        {
            const auto cos_dec = aten::cos(dec);

            return aten::normalize(aten::vec3(
                cos_dec * aten::cos(ra),
                aten::sin(dec),
                cos_dec * aten::sin(ra)));
        }

        template <class T>
        T ReadValue(const std::array<uint8_t, 32>& record, const size_t offset, const bool swap_endian)
        {
            // Avoid casting the byte buffer to a packed struct. This keeps the code
            // independent of compiler packing/alignment rules.
            T value{};
            std::memcpy(&value, record.data() + offset, sizeof(T));

            if (swap_endian) {
                auto* bytes = reinterpret_cast<uint8_t*>(&value);
                std::reverse(bytes, bytes + sizeof(T));
            }

            return value;
        }

        int32_t SwapEndian(const int32_t value)
        {
            auto swapped = value;
            auto* bytes = reinterpret_cast<uint8_t*>(&swapped);
            std::reverse(bytes, bytes + sizeof(int32_t));
            return swapped;
        }

        bool ParseBrightStarCatalogLine(std::string_view line, Star& star)
        {
            // Fixed-width ASCII fields used here:
            //   HR       bytes 1-4
            //   RAJ2000  bytes 76-83  RAh/RAm/RAs
            //   DEJ2000  bytes 84-90  sign/deg/arcmin/arcsec
            //   Vmag     bytes 103-107
            //   B-V      bytes 110-114
            //   pmRA     bytes 149-154
            //   pmDE     bytes 155-160
            int32_t hr = 0;
            if (!ParseInt(Field(line, 1, 4), hr)) {
                return false;
            }

            int32_t ra_h = 0;
            int32_t ra_m = 0;
            float ra_s = 0.0F;
            int32_t dec_deg = 0;
            int32_t dec_min = 0;
            int32_t dec_sec = 0;
            float visual_magnitude = 0.0F;

            if (!ParseInt(Field(line, 76, 77), ra_h)
                || !ParseInt(Field(line, 78, 79), ra_m)
                || !ParseFloat(Field(line, 80, 83), ra_s)
                || !ParseInt(Field(line, 85, 86), dec_deg)
                || !ParseInt(Field(line, 87, 88), dec_min)
                || !ParseInt(Field(line, 89, 90), dec_sec)
                || !ParseFloat(Field(line, 103, 107), visual_magnitude)) {
                return false;
            }

            const auto dec_sign_field = Field(line, 84, 84);
            const auto dec_sign = dec_sign_field == "-" ? -1 : 1;

            float bv_color = 0.0F;
            // B-V and proper motion fields can be blank for some records. Keep
            // zero defaults so a valid star is still loaded.
            ParseFloat(Field(line, 110, 114), bv_color);

            float proper_motion_ra = 0.0F;
            float proper_motion_dec = 0.0F;
            ParseFloat(Field(line, 149, 154), proper_motion_ra);
            ParseFloat(Field(line, 155, 160), proper_motion_dec);

            star.hr = static_cast<uint32_t>(hr);
            star.direction_j2000 = ComputeDirectionFromRaDec(ra_h, ra_m, ra_s, dec_sign, dec_deg, dec_min, dec_sec);
            star.visual_magnitude = visual_magnitude;
            star.irradiance = ComputeStarIrradianceFromMagnitude(visual_magnitude);
            star.bv_color = bv_color;
            star.proper_motion_ra = proper_motion_ra;
            star.proper_motion_dec = proper_motion_dec;
            star.color = ComputeBlackbodyColor(ComputeEffectiveTemperatureFromBv(bv_color));

            return true;
        }

        bool ParseBrightStarCatalogBinaryRecord(
            const std::array<uint8_t, 32>& record,
            const bool swap_endian,
            Star& star)
        {
            // Common BSC5 binary record layout:
            //   float  catalog number
            //   double right ascension, radians, J2000
            //   double declination, radians, J2000
            //   char[2] spectral type shorthand
            //   int16  visual magnitude * 100
            //   float  proper motion RA, radians/year
            //   float  proper motion Dec, radians/year
            // It does not contain B-V, so color is later estimated from the
            // two-character spectral type.
            constexpr auto ArcsecPerRadian = 206264.80624709636F;

            const auto catalog_number = ReadValue<float>(record, 0, swap_endian);
            const auto ra = static_cast<float>(ReadValue<double>(record, 4, swap_endian));
            const auto dec = static_cast<float>(ReadValue<double>(record, 12, swap_endian));

            std::string spectral_type;
            spectral_type.push_back(static_cast<char>(record[20]));
            spectral_type.push_back(static_cast<char>(record[21]));
            spectral_type = Trim(spectral_type);

            const auto magnitude_x100 = ReadValue<int16_t>(record, 22, swap_endian);
            const auto proper_motion_ra = ReadValue<float>(record, 24, swap_endian);
            const auto proper_motion_dec = ReadValue<float>(record, 28, swap_endian);

            const auto visual_magnitude = static_cast<float>(magnitude_x100) * 0.01F;
            const auto bv_color = EstimateBvFromSpectralType(spectral_type);

            // catalog_number is stored as float in the original binary layout.
            // It is an integer-valued catalog id, so truncating to uint32_t is OK.
            star.hr = static_cast<uint32_t>(aten::max(0.0F, catalog_number));
            star.direction_j2000 = ComputeDirectionFromRaDecRadians(ra, dec);
            star.visual_magnitude = visual_magnitude;
            star.irradiance = ComputeStarIrradianceFromMagnitude(visual_magnitude);
            star.bv_color = bv_color;

            // Convert binary proper motion from radians/year into the same
            // arcsec/year representation used by the ASCII loader.
            star.proper_motion_ra = proper_motion_ra * ArcsecPerRadian;
            star.proper_motion_dec = proper_motion_dec * ArcsecPerRadian;
            star.color = ComputeBlackbodyColor(ComputeEffectiveTemperatureFromBv(bv_color));

            return star.hr > 0;
        }
    }

    bool StarCatalog::LoadFromBrightStarCatalog(const std::string& path)
    {
        return LoadFromBrightStarCatalogAscii(path);
    }

    bool StarCatalog::LoadFromBrightStarCatalogAscii(const std::string& path)
    {
        // ASCII path: CDS/VizieR fixed-width text catalog. This is slower than
        // binary but easier to inspect and includes B-V directly.
        std::ifstream ifs(path);
        if (!ifs) {
            return false;
        }

        std::vector<Star> stars;

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                continue;
            }

            Star star;
            if (ParseBrightStarCatalogLine(line, star)) {
                stars.push_back(star);
            }
        }

        stars_ = std::move(stars);
        return true;
    }

    bool StarCatalog::LoadFromBrightStarCatalogBinary(const std::string& path)
    {
        // Binary path: common BSC5 binary distribution. This is faster to load
        // but stores less color information than the fixed-width ASCII catalog.
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            return false;
        }

        std::array<int32_t, 7> header{};
        if (!ifs.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(sizeof(int32_t) * header.size()))) {
            return false;
        }

        // Header layout used by the common BSC5 binary file:
        // STAR0, STAR1, STARN, STNUM, MPROP, NMAG, NBENT.
        // STARN can be negative in some distributions; the sign is metadata,
        // while abs(STARN) is the number of records.
        auto star_count = header[2];
        auto has_star_number = header[3];
        auto has_proper_motion = header[4];
        auto magnitude_count = header[5];
        auto bytes_per_entry = header[6];
        bool swap_endian = false;

        if (aten::abs(star_count) > 1000000 || bytes_per_entry <= 0 || bytes_per_entry > 1024) {
            // If the header is obviously unreasonable, try the opposite endian.
            swap_endian = true;
            for (auto& value : header) {
                value = SwapEndian(value);
            }

            star_count = header[2];
            has_star_number = header[3];
            has_proper_motion = header[4];
            magnitude_count = header[5];
            bytes_per_entry = header[6];
        }

        star_count = aten::abs(star_count);

        if (star_count <= 0
            || bytes_per_entry != 32
            || has_star_number == 0
            || has_proper_motion == 0
            || magnitude_count <= 0) {
            // This loader intentionally supports only the full 32-byte BSC5
            // record layout used by the common binary catalog file.
            return false;
        }

        std::vector<Star> stars;
        stars.reserve(static_cast<size_t>(star_count));

        std::vector<uint8_t> raw_record(static_cast<size_t>(bytes_per_entry));
        for (int32_t i = 0; i < star_count; ++i) {
            if (!ifs.read(reinterpret_cast<char*>(raw_record.data()), bytes_per_entry)) {
                return false;
            }

            std::array<uint8_t, 32> record{};
            std::copy(raw_record.begin(), raw_record.begin() + 32, record.begin());

            Star star;
            if (ParseBrightStarCatalogBinaryRecord(record, swap_endian, star)) {
                stars.push_back(star);
            }
        }

        stars_ = std::move(stars);
        return true;
    }
}
