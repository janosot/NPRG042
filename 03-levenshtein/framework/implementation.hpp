#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG> {
private:
    std::vector<DIST> mCol;
    std::vector<DIST> mRow;
    std::vector<DIST> mDiag;
    static constexpr std::size_t block_size = 64;

    void block_process(std::size_t row_index, std::size_t col_index, std::size_t diag_index, const std::vector<C> &str1, const std::vector<C> &str2)
    {
        std::size_t start_row_index = row_index * block_size + 1;
        std::size_t start_col_index = col_index * block_size + 1;
        DIST diags = mDiag[diag_index];
#pragma omp simd
        for (std::size_t row_i = start_row_index; row_i < start_row_index + block_size; ++row_i) 
		{
            DIST lastUpper = mCol[row_i];
            for (std::size_t col_i = start_col_index; col_i < start_col_index + block_size; ++col_i) 
			{
                DIST dist1 = std::min<DIST>(mRow[col_i], mCol[row_i]) + 1;
                DIST dist2 = diags + (str1[col_i - 1] == str2[row_i - 1] ? 0 : 1);
                diags = mRow[col_i];
                mRow[col_i] = std::min<DIST>(dist1, dist2);
				mCol[row_i] = std::min<DIST>(dist1, dist2);
            }
            diags = lastUpper;
        }
        mDiag[diag_index] = diags;
    }

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		if (len1 < len2)
		{
			std::swap(len1, len2);
		}
		
		mCol.resize((std::size_t)len1 + 1);
		mRow.resize((std::size_t)len2 + 1);
		mDiag.resize((std::size_t)len2 + 1);

		for (std::size_t i = 0; i < mCol.size(); ++i) 
		{
			mCol[i] = i + 1;
		}
		for (std::size_t i = 0; i < mRow.size(); ++i) 
		{
			mRow[i] = i + 1;
		}

		mDiag[0] = 0;
	}


	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		std::size_t len1 = str1.size();
		std::size_t len2 = str2.size();
        assert(len1 % block_size == 0);
        assert(len2 % block_size == 0);
		if (len1 < len2)
		{
			std::swap(len1, len2);
		}
		
        // Special case (one of the strings is empty).
        if (len1 == 0 || len2 == 0) 
		{
			return std::max<DIST>(len1, len2);
		}

        std::vector<C> *s1 = &(const_cast<std::vector<C> &>(str1));
        std::vector<C> *s2 = &(const_cast<std::vector<C> &>(str2));
	    if (s1->size() > s2->size()) 
		{
	        std::swap(s1, s2);
	    }

		const std::size_t rows = s2->size() / block_size;
        const std::size_t cols = s1->size() / block_size;
        for (std::size_t i = 1; i < rows + cols; ++i) 
		{
#pragma omp parallel for schedule(static)
            for (std::size_t col_i = (i < rows) ? 0 : i - rows; col_i <= std::min(i - 1, cols - 1); ++col_i) 
			{
                std::size_t row_i = i - col_i - 1;
                block_process(row_i, col_i, col_i, *s1, *s2);
            }

            if (i < rows) 
			{
                mDiag[i] = i * block_size;
            }
        }

        return mRow.back();
	}
};

#endif