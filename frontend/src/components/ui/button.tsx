import { forwardRef, type ButtonHTMLAttributes } from 'react'

type ButtonVariant = 'default' | 'outline' | 'ghost' | 'danger'

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant
}

const variants: Record<ButtonVariant, string> = {
  default: 'button button-primary',
  outline: 'button button-outline',
  ghost: 'button button-ghost',
  danger: 'button button-danger',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className = '', variant = 'default', ...props },
  ref,
) {
  return <button ref={ref} className={`${variants[variant]} ${className}`} {...props} />
})
