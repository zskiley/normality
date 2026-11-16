// The one translation unit that uses the external equivalence library
//   https://github.com/zskiley/Affine-and-linear-equivalence-algorithms
// Compile as C++20 with the library's src directory on the include path, and
// link with its dfs.cpp and equivalence_api.cpp (see README.md).
#include "equivalence_api.hpp"

#include <cstdint>
#include <exception>
#include <vector>

// table[x] = f(x) on m variables.  Generators of the group of affine maps A
// with f(Ax) = f(x) or f(Ax) = f(x) + 1: each as the m images of the unit
// vectors followed by the image of 0.  `order` is the order of that group.
bool aleq_generators(const std::vector<std::uint32_t>& table, int m,
                     std::vector<std::vector<std::uint32_t>>& gens, double& order)
{
    try {
        auto eq = affine::api::generate_self_equivalences(
            static_cast<std::uint32_t>(m), 1u, table);
        affine::group::SchreierSims chain;
        chain.reset(static_cast<std::uint32_t>(m));
        for (const auto& g : eq.paired_group_generators()) {
            const affine::f2::AffineMap& a = g.domain_map;
            if (a.basis_images.size() != static_cast<std::size_t>(m)) {
                return false;
            }
            std::vector<std::uint32_t> row(a.basis_images.begin(), a.basis_images.end());
            row.push_back(a.translation);
            gens.push_back(std::move(row));
            (void)chain.add_generator(a);
        }
        order = 1.0;
        for (const std::size_t s : chain.orbit_sizes()) {
            order *= static_cast<double>(s);
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
