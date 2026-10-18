import { describe, expect, it } from 'vitest'
import { EMPTY_POSTING, SAMPLE_POSTING } from '../constants/form'
import { validatePosting } from './JobPostingForm'

describe('validatePosting', () => {
  it('requires all four categorical fields expected by the backend', () => {
    const errors = validatePosting(EMPTY_POSTING)
    expect(errors.department).toMatch(/required/i)
    expect(errors.employment_type).toMatch(/required/i)
    expect(errors.required_experience).toMatch(/required/i)
    expect(errors.required_education).toMatch(/required/i)
  })

  it('accepts the illustrative backend-compatible sample', () => {
    expect(validatePosting(SAMPLE_POSTING)).toEqual({})
  })
})
